#include "volrend/common.hpp"

// Shader backend only enabled when build with VOLREND_USE_CUDA=OFF
#ifndef VOLREND_CUDA
#include "volrend/renderer.hpp"
#include <glm/gtc/type_ptr.hpp>

#include <fstream>
#include <cstdint>
#include <string>

#ifdef __EMSCRIPTEN__
// WebGL
#include <GLES3/gl3.h>
#else
#include <GL/glew.h>
#endif

#include "volrend/internal/rt_frag.inl"
#include "volrend/internal/shader.hpp"

namespace volrend {

namespace {

const char* PASSTHRU_VERT_SHADER_SRC =
    R"glsl(#version 300 es
layout (location = 0) in vec3 aPos;

void main()
{
    gl_Position = vec4(aPos.x, aPos.y, aPos.z, 1.0);
}
)glsl";

const float quad_verts[] = {
    -1.f, -1.f, 0.5f, 1.f, -1.f, 0.5f, -1.f, 1.f, 0.5f, 1.f, 1.f, 0.5f,
};

struct _RenderUniforms {
    GLint cam_transform, cam_focal, cam_reso;
    GLint opt_step_size, opt_backgrond_brightness, opt_stop_thresh,
        opt_sigma_thresh, opt_render_bbox, opt_basis_minmax, opt_rot_dirs;
    GLint tree_data_tex, tree_child_tex, tree_extra_tex;
};

}  // namespace

struct VolumeRenderer::Impl {
    Impl(Camera& camera, RenderOptions& options, int max_tries = 4)
        : camera(camera), options(options) {}

    ~Impl() {
        glDeleteProgram(program);
        glDeleteTextures(1, &tex_tree_data);
        glDeleteTextures(1, &tex_tree_child);
        glDeleteTextures(1, &tex_tree_extra);
    }

    void start() {
        if (started_) return;
        resize(0, 0);

        glGetIntegerv(GL_MAX_TEXTURE_SIZE, &tex_max_size);
        // int tex_3d_max_size;
        // glGetIntegerv(GL_MAX_3D_TEXTURE_SIZE, &tex_3d_max_size);
        // std::cout << " texture dim limit: " << tex_max_size << "\n";
        // std::cout << " texture 3D dim limit: " << tex_3d_max_size << "\n";

        glGenTextures(1, &tex_tree_data);
        glGenTextures(1, &tex_tree_child);
        glGenTextures(1, &tex_tree_extra);

        quad_init();
        shader_init();
        started_ = true;
    }

    void render() {
        glClear(GL_COLOR_BUFFER_BIT);
        if (tree == nullptr || !started_) return;

        camera._update();
        // FIXME reduce uniform transfers?
        glUniformMatrix4x3fv(u.cam_transform, 1, GL_FALSE,
                             glm::value_ptr(camera.transform));
        glUniform2f(u.cam_focal, camera.fx, camera.fy);
        glUniform2f(u.cam_reso, (float)camera.width, (float)camera.height);
        glUniform1f(u.opt_step_size, options.step_size);
        glUniform1f(u.opt_backgrond_brightness, options.background_brightness);
        glUniform1f(u.opt_stop_thresh, options.stop_thresh);
        glUniform1f(u.opt_sigma_thresh, options.sigma_thresh);
        glUniform1fv(u.opt_render_bbox, 6, options.render_bbox);
        glUniform1iv(u.opt_basis_minmax, 2, options.basis_minmax);
        glUniform3fv(u.opt_rot_dirs, 1, options.rot_dirs);

        // FIXME Probably can be done ony once
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, tex_tree_child);

        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, tex_tree_data);

        // glActiveTexture(GL_TEXTURE2);
        // glBindTexture(GL_TEXTURE_2D, tex_tree_extra);

        glBindVertexArray(vao_quad);
        glDrawArrays(GL_TRIANGLE_STRIP, 0, (GLsizei)4);
        // glBindVertexArray(0);
    }

    void set(N3Tree& tree) {
        start();
        if (tree.capacity > 0) {
            this->tree = &tree;
            upload_data();
            upload_child_links();
            upload_tree_spec();
        }
    }

    void clear() { this->tree = nullptr; }

    void resize(const int width, const int height) {
        if (camera.width == width && camera.height == height) return;
        if (width > 0) {
            camera.width = width;
            camera.height = height;
        }
        glViewport(0, 0, width, height);
    }

    N3Tree* tree;

   private:
    void auto_size_2d(size_t size, size_t& width, size_t& height,
                      int base_dim = 1) {
        if (size == 0) {
            width = height = 0;
            return;
        }
        width = std::sqrt(size);
        if (width % base_dim) {
            width += base_dim - width % base_dim;
        }
        height = (size - 1) / width + 1;
        if (height > tex_max_size || width > tex_max_size) {
            throw std::runtime_error(
                "Octree data exceeds hardward 2D texture limit\n");
        }
    }

    void upload_data() {
        const GLint data_size =
            tree->capacity * tree->N * tree->N * tree->N * tree->data_dim;
        size_t width, height;
        auto_size_2d(data_size, width, height, tree->data_dim);
        // FIXME can we remove the copy to float here?
        // Can't seem to get half glTexImage2D to work
        const size_t pad = width * height - data_size;
        tree->data_.data_holder.resize((data_size + pad) * sizeof(half));
        glUniform1i(glGetUniformLocation(program, "tree_data_dim"), width);

        glBindTexture(GL_TEXTURE_2D, tex_tree_data);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_R16F, width, height, 0, GL_RED,
                     GL_HALF_FLOAT, tree->data_.data<half>());
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

        // Maybe upload extra data
        const size_t extra_sz = tree->extra_.data_holder.size() / sizeof(float);
        if (extra_sz) {
            glBindTexture(GL_TEXTURE_2D, tex_tree_extra);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F,
                         extra_sz / tree->data_format.basis_dim,
                         tree->data_format.basis_dim, 0, GL_RED, GL_FLOAT,
                         tree->extra_.data<float>());
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        }

        glBindTexture(GL_TEXTURE_2D, 0);
    }

    void upload_child_links() {
        const size_t child_size =
            size_t(tree->capacity) * tree->N * tree->N * tree->N;
        size_t width, height;
        auto_size_2d(child_size, width, height);

        const size_t pad = width * height - child_size;
        tree->child_.data_holder.resize((child_size + pad) * sizeof(int32_t));
        glUniform1i(glGetUniformLocation(program, "tree_child_dim"), width);

        glBindTexture(GL_TEXTURE_2D, tex_tree_child);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_R32I, width, height, 0,
                     GL_RED_INTEGER, GL_INT, tree->child_.data<int32_t>());
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

        glBindTexture(GL_TEXTURE_2D, 0);
    }

    void upload_tree_spec() {
        glUniform1i(glGetUniformLocation(program, "tree.N"), tree->N);
        glUniform1i(glGetUniformLocation(program, "tree.data_dim"),
                    tree->data_dim);
        glUniform1i(glGetUniformLocation(program, "tree.format"),
                    (int)tree->data_format.format);
        glUniform1i(glGetUniformLocation(program, "tree.basis_dim"),
                    tree->data_format.basis_dim);
        glUniform3f(glGetUniformLocation(program, "tree.center"),
                    tree->offset[0], tree->offset[1], tree->offset[2]);
        glUniform3f(glGetUniformLocation(program, "tree.scale"), tree->scale[0],
                    tree->scale[1], tree->scale[2]);
        if (tree->use_ndc) {
            glUniform1f(glGetUniformLocation(program, "tree.ndc_width"),
                        tree->ndc_width);
            glUniform1f(glGetUniformLocation(program, "tree.ndc_height"),
                        tree->ndc_height);
            glUniform1f(glGetUniformLocation(program, "tree.ndc_focal"),
                        tree->ndc_focal);
        } else {
            glUniform1f(glGetUniformLocation(program, "tree.ndc_width"), -1.f);
        }
    }

    void shader_init() {
        program = create_shader_program(PASSTHRU_VERT_SHADER_SRC, RT_FRAG_SRC);

        u.cam_transform = glGetUniformLocation(program, "cam.transform");
        u.cam_focal = glGetUniformLocation(program, "cam.focal");
        u.cam_reso = glGetUniformLocation(program, "cam.reso");
        u.opt_step_size = glGetUniformLocation(program, "opt.step_size");
        u.opt_backgrond_brightness =
            glGetUniformLocation(program, "opt.background_brightness");
        u.opt_stop_thresh = glGetUniformLocation(program, "opt.stop_thresh");
        u.opt_sigma_thresh = glGetUniformLocation(program, "opt.sigma_thresh");
        u.opt_render_bbox = glGetUniformLocation(program, "opt.render_bbox");
        u.opt_basis_minmax = glGetUniformLocation(program, "opt.basis_minmax");
        u.opt_rot_dirs = glGetUniformLocation(program, "opt.rot_dirs");
        u.tree_data_tex = glGetUniformLocation(program, "tree_data_tex");
        u.tree_child_tex = glGetUniformLocation(program, "tree_child_tex");
        u.tree_extra_tex = glGetUniformLocation(program, "tree_extra_tex");
        glUniform1i(u.tree_child_tex, 0);
        glUniform1i(u.tree_data_tex, 1);
        glUniform1i(u.tree_extra_tex, 2);
    }

    void quad_init() {
        GLuint vbo;
        glGenBuffers(1, &vbo);
        glGenVertexArrays(1, &vao_quad);
        glBindVertexArray(vao_quad);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, sizeof quad_verts, (GLvoid*)quad_verts,
                     GL_STATIC_DRAW);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float),
                              (GLvoid*)0);
        glEnableVertexAttribArray(0);
        glBindVertexArray(0);
    }

    Camera& camera;
    RenderOptions& options;

    GLuint program = -1;
    GLuint tex_tree_data = -1, tex_tree_child, tex_tree_extra;
    GLuint vao_quad;
    GLint tex_max_size;

    std::string shader_fname = "shaders/rt.frag";

    _RenderUniforms u;
    bool started_ = false;
};

VolumeRenderer::VolumeRenderer()
    : impl_(std::make_unique<Impl>(camera, options)) {}

VolumeRenderer::~VolumeRenderer() {}

void VolumeRenderer::render() { impl_->render(); }

void VolumeRenderer::set(N3Tree& tree) { impl_->set(tree); }
const N3Tree& VolumeRenderer::get() const { return *impl_->tree; }
void VolumeRenderer::clear() { impl_->clear(); }

void VolumeRenderer::resize(int width, int height) {
    impl_->resize(width, height);
}
const char* VolumeRenderer::get_backend() { return "Shader"; }

}  // namespace volrend

#endif
