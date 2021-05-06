#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_inverse.hpp>

#include "volrend/cuda/common.cuh"

#include <cstdlib>
#include <cstdio>
#include <chrono>
#include <string>
#include <fstream>

#include "volrend/renderer.hpp"
#include "volrend/n3tree.hpp"

#include "volrend/internal/opts.hpp"
#include "volrend/internal/imwrite.hpp"

#include "imgui_impl_opengl3.h"
#include "imgui_impl_glfw.h"

#include "ImGuizmo.h"

#ifndef __EMSCRIPTEN__
#include "imfilebrowser.h"
#endif

#ifdef VOLREND_CUDA
#include "volrend/cuda/common.cuh"
#endif

namespace volrend {

namespace {

#define GET_RENDERER(window) \
    (*((VolumeRenderer*)glfwGetWindowUserPointer(window)))

void glfw_update_title(GLFWwindow* window) {
    // static fps counters
    // Source: http://antongerdelan.net/opengl/glcontext2.html
    static double stamp_prev = 0.0;
    static int frame_count = 0;

    const double stamp_curr = glfwGetTime();
    const double elapsed = stamp_curr - stamp_prev;

    if (elapsed > 0.5) {
        stamp_prev = stamp_curr;

        const double fps = (double)frame_count / elapsed;

        char tmp[128];
        sprintf(tmp, "plenoctree viewer - FPS: %.2f", fps);
        glfwSetWindowTitle(window, tmp);
        frame_count = 0;
    }

    frame_count++;
}

void glfw_error_callback(int error, const char* description) {
    fputs(description, stderr);
}

void glfw_key_callback(GLFWwindow* window, int key, int scancode, int action,
                       int mods) {
    ImGui_ImplGlfw_KeyCallback(window, key, scancode, action, mods);
    if (ImGui::GetIO().WantCaptureKeyboard) return;

    if (action == GLFW_PRESS || action == GLFW_REPEAT) {
        auto& rend = GET_RENDERER(window);
        auto& cam = rend.camera;
        switch (key) {
            case GLFW_KEY_ESCAPE:
                glfwSetWindowShouldClose(window, GL_TRUE);
                break;
            case GLFW_KEY_W:
            case GLFW_KEY_S:
            case GLFW_KEY_A:
            case GLFW_KEY_D:
            case GLFW_KEY_E:
            case GLFW_KEY_Q: {
                // Camera movement
                float speed = 0.002f;
                if (mods & GLFW_MOD_SHIFT) speed *= 5.f;
                if (key == GLFW_KEY_S || key == GLFW_KEY_A || key == GLFW_KEY_E)
                    speed = -speed;
                const auto& vec =
                    (key == GLFW_KEY_A || key == GLFW_KEY_D)   ? cam.v_right
                    : (key == GLFW_KEY_W || key == GLFW_KEY_S) ? -cam.v_back
                                                               : -cam.v_up;
                cam.move(vec * speed);
            } break;

            case GLFW_KEY_C: {
                // Print C2W matrix
                std::cout << "C2W:\n";
                for (int i = 0; i < 3; ++i) {
                    for (int j = 0; j < 4; ++j) {
                        if (j) std::cout << " ";
                        std::cout << cam.transform[j][i];
                    }
                    std::cout << "\n";
                }
                std::flush(std::cout);
            } break;

            case GLFW_KEY_P: {
                // Print pose
                std::cout << "Pose:\n";
                const N3Tree& tree = rend.get();
                for (int i = 0; i < tree.pose.size(); ++i) {
                    for (int j = 0; j < 3; ++j) {
                        if (j) std::cout << " ";
                        std::cout << tree.pose[i][j];
                    }
                    std::cout << "\n";
                }
                std::flush(std::cout);
            } break;

#ifdef VOLREND_CUDA
            case GLFW_KEY_I:
            case GLFW_KEY_J:
            case GLFW_KEY_K:
            case GLFW_KEY_L:
            case GLFW_KEY_U:
            case GLFW_KEY_O:
                if (rend.options.enable_probe) {
                    // Probe movement
                    float speed = 0.002f;
                    if (mods & GLFW_MOD_SHIFT) speed *= 5.f;
                    if (key == GLFW_KEY_J || key == GLFW_KEY_K ||
                        key == GLFW_KEY_U)
                        speed = -speed;
                    int dim = (key == GLFW_KEY_J || key == GLFW_KEY_L)   ? 0
                              : (key == GLFW_KEY_I || key == GLFW_KEY_K) ? 1
                                                                         : 2;
                    rend.options.probe[dim] += speed;
                }
                break;
#endif

            case GLFW_KEY_MINUS:
                cam.fx *= 0.99f;
                cam.fy *= 0.99f;
                break;

            case GLFW_KEY_EQUAL:
                cam.fx *= 1.01f;
                cam.fy *= 1.01f;
                break;

            case GLFW_KEY_0:
                cam.fx = CAMERA_DEFAULT_FOCAL_LENGTH;
                cam.fy = CAMERA_DEFAULT_FOCAL_LENGTH;
                break;

            case GLFW_KEY_1:
                cam.v_world_up = glm::vec3(0.f, 0.f, 1.f);
                break;

            case GLFW_KEY_2:
                cam.v_world_up = glm::vec3(0.f, 0.f, -1.f);
                break;

            case GLFW_KEY_3:
                cam.v_world_up = glm::vec3(0.f, 1.f, 0.f);
                break;

            case GLFW_KEY_4:
                cam.v_world_up = glm::vec3(0.f, -1.f, 0.f);
                break;

            case GLFW_KEY_5:
                cam.v_world_up = glm::vec3(1.f, 0.f, 0.f);
                break;

            case GLFW_KEY_6:
                cam.v_world_up = glm::vec3(-1.f, 0.f, 0.f);
                break;
        }
    }
}

void glfw_mouse_button_callback(GLFWwindow* window, int button, int action,
                                int mods) {
    ImGui_ImplGlfw_MouseButtonCallback(window, button, action, mods);
    if (ImGui::GetIO().WantCaptureMouse) return;

    auto& rend = GET_RENDERER(window);
    auto& cam = rend.camera;
    double x, y;
    glfwGetCursorPos(window, &x, &y);
    if (action == GLFW_PRESS) {
        if (button == GLFW_MOUSE_BUTTON_LEFT) {
            const N3Tree& tree = rend.get();
            glm::mat4 w2c = glm::affineInverse(glm::mat4(cam.transform));
            glm::mat4 camera_persp_prj(cam.fx / cam.width * 2.0, 0.f, 0.f, 0.f,
                                       0.f, cam.fy / cam.height * 2.0, 0.f, 0.f,
                                       0.f, 0.f, -1.0002f, -1.f, 0.f, 0.f,
                                       -0.02f, 0.f);

            glm::mat4 VP = camera_persp_prj * w2c;
            glm::vec2 mouse(x, y);

            const float SELECT_SCREEN_THRESH =
                (rend.options.show_joints ? 5.f : 15.f) / 800.f * cam.width;

            float min_z = 1e9f;
            int min_joint = -1;
            for (int i = 0; i < tree.n_joints; ++i) {
                glm::vec4 pix = VP * glm::vec4(tree.joint_pos_posed_[i], 1.f);
                pix /= pix[3];
                pix[0] = (pix[0] + 1.f) * cam.width * 0.5f;
                pix[1] = (-pix[1] + 1.f) * cam.height * 0.5f;

                float dist = glm::length(glm::vec2(pix) - mouse);
                if (dist < SELECT_SCREEN_THRESH && pix[2] < min_z) {
                    min_z = pix[2];
                    min_joint = i;
                }
            }

            if (~min_joint) {
                rend.options.selected_joint = min_joint;
            } else {
                rend.options.selected_joint = -2;
            }
        } else {
            cam.begin_drag(
                x, y,
                (mods & GLFW_MOD_SHIFT) || button == GLFW_MOUSE_BUTTON_MIDDLE,
                button == GLFW_MOUSE_BUTTON_RIGHT);
        }
    } else if (action == GLFW_RELEASE) {
        cam.end_drag();
    }
}

void glfw_cursor_pos_callback(GLFWwindow* window, double x, double y) {
    GET_RENDERER(window).camera.drag_update(x, y);
}

void glfw_scroll_callback(GLFWwindow* window, double xoffset, double yoffset) {
    ImGui_ImplGlfw_ScrollCallback(window, xoffset, yoffset);
    if (ImGui::GetIO().WantCaptureMouse) return;
    auto& cam = GET_RENDERER(window).camera;
    // Focal length adjusting was very annoying so changed it to movement in
    // z cam.focal *= (yoffset > 0.f) ? 1.01f : 0.99f;
    const float speed_fact = 1e-1f;
    cam.move(cam.v_back * ((yoffset < 0.f) ? speed_fact : -speed_fact));
}

GLFWwindow* glfw_init(const int width, const int height) {
    glfwSetErrorCallback(glfw_error_callback);

    if (!glfwInit()) std::exit(EXIT_FAILURE);

    glfwWindowHint(GLFW_DEPTH_BITS, GL_TRUE);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);

    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window =
        glfwCreateWindow(width, height, "volrend viewer", NULL, NULL);

    glClearDepth(1.0);
    glDepthFunc(GL_LESS);
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

    if (window == nullptr) {
        glfwTerminate();
        std::exit(EXIT_FAILURE);
    }

    glfwMakeContextCurrent(window);

    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
        fputs("GLEW init failed\n", stderr);
        getchar();
        glfwTerminate();
        std::exit(EXIT_FAILURE);
    }

    // ignore vsync for now
    glfwSwapInterval(0);

    // only copy r/g/b
    glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_FALSE);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();

    ImGui_ImplGlfw_InitForOpenGL(window, false);
    char* glsl_version = NULL;
    ImGui_ImplOpenGL3_Init(glsl_version);
    // Setup Dear ImGui style
    ImGui::StyleColorsDark();
    ImGui::GetIO().IniFilename = nullptr;
    glfwSetCharCallback(window, ImGui_ImplGlfw_CharCallback);

    return window;
}

void glfw_window_size_callback(GLFWwindow* window, int width, int height) {
    glViewport(0, 0, width, height);
    GET_RENDERER(window).resize(width, height);
}

void draw_imgui(VolumeRenderer& rend, N3Tree& tree,
                const std::vector<std::string>& joint_names) {
    auto& cam = rend.camera;
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    // clang-format off
    static glm::mat4 camera_persp_prj(1.f, 0.f, 0.f, 0.f,
                                         0.f, 1.f, 0.f, 0.f,
                                         0.f, 0.f, -1.f, -1.f,
                                         0.f, 0.f, -0.001f, 0.f);
    // clang-format on
    ImGuiIO& io = ImGui::GetIO();

    camera_persp_prj[0][0] = cam.fx / cam.width * 2.0;
    camera_persp_prj[1][1] = cam.fy / cam.height * 2.0;
    ImGuizmo::SetOrthographic(false);
    ImGuizmo::SetGizmoSizeClipSpace(0.05f);

    ImGuizmo::BeginFrame();

    ImGuizmo::SetRect(0, 0, io.DisplaySize.x, io.DisplaySize.y);
    glm::mat4 w2c = glm::affineInverse(glm::mat4(cam.transform));

    ImGui::SetNextWindowPos(ImVec2(20.f, 20.f), ImGuiCond_Once);
    ImGui::SetNextWindowSize(ImVec2(340.f, 480.f), ImGuiCond_Once);

    static char title[128] = {0};
    if (title[0] == 0) {
        sprintf(title, "volrend backend: %s", rend.get_backend());
    }

    // Begin window
    ImGui::Begin(title);
#ifndef __EMSCRIPTEN__
#ifdef VOLREND_CUDA
    static ImGui::FileBrowser open_obj_mesh_dialog(
        ImGuiFileBrowserFlags_MultipleSelection);
    static ImGui::FileBrowser open_animation_dialog;
    if (open_obj_mesh_dialog.GetTitle().empty()) {
        open_obj_mesh_dialog.SetTypeFilters({".obj"});
        open_obj_mesh_dialog.SetTitle("Load basic triangle OBJ");
    }
    if (open_animation_dialog.GetTitle().empty()) {
        open_animation_dialog.SetTypeFilters({".npy"});
        open_animation_dialog.SetTitle(
            "Load animation sequence (#frames x #joints x 3 matrix)");
    }
#endif
    // static ImGui::FileBrowser open_tree_dialog;
    // if (open_tree_dialog.GetTitle().empty()) {
    //     open_tree_dialog.SetTypeFilters({".npz"});
    //     open_tree_dialog.SetTitle("Load N3Tree npz from svox");
    // }
    static ImGui::FileBrowser save_screenshot_dialog(
        ImGuiFileBrowserFlags_EnterNewFilename);
    if (save_screenshot_dialog.GetTitle().empty()) {
        save_screenshot_dialog.SetTypeFilters({".png"});
        save_screenshot_dialog.SetTitle("Save screenshot (png)");
    }

    // if (ImGui::Button("Open Tree")) {
    //     open_tree_dialog.Open();
    // }
    // ImGui::SameLine();
    if (ImGui::Button("Save Screenshot")) {
        save_screenshot_dialog.Open();
    }

    // open_tree_dialog.Display();
    // if (open_tree_dialog.HasSelected()) {
    //     // Load octree
    //     std::string path = open_tree_dialog.GetSelected().string();
    //     std::cout << "Load N3Tree npz: " << path << "\n";
    //     tree.open(path);
    //     rend.set(tree);
    //     open_tree_dialog.ClearSelected();
    // }

    save_screenshot_dialog.Display();
    if (save_screenshot_dialog.HasSelected()) {
        // Save screenshot
        std::string path = save_screenshot_dialog.GetSelected().string();
        save_screenshot_dialog.ClearSelected();
        int width = rend.camera.width, height = rend.camera.height;
        std::vector<unsigned char> windowPixels(4 * width * height);
        glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE,
                     &windowPixels[0]);

        std::vector<unsigned char> flippedPixels(4 * width * height);
        for (int row = 0; row < height; ++row)
            memcpy(&flippedPixels[row * width * 4],
                   &windowPixels[(height - row - 1) * width * 4], 4 * width);

        if (path.size() < 4 ||
            path.compare(path.size() - 4, 4, ".png", 0, 4) != 0) {
            path.append(".png");
        }
        if (internal::write_png_file(path, flippedPixels.data(), width,
                                     height)) {
            std::cout << "Wrote " << path << "\n";
        } else {
            std::cout << "Failed to save screenshot\n";
        }
    }
#endif

    ImGui::SetNextTreeNodeOpen(false, ImGuiCond_Once);
    if (ImGui::CollapsingHeader("Camera")) {
        // Update vectors indirectly since we need to normalize on change
        // (press update button) and it would be too confusing to keep
        // normalizing
        static glm::vec3 world_up_tmp = rend.camera.v_world_up;
        static glm::vec3 world_down_prev = rend.camera.v_world_up;
        static glm::vec3 back_tmp = rend.camera.v_back;
        static glm::vec3 forward_prev = rend.camera.v_back;
        if (cam.v_world_up != world_down_prev)
            world_up_tmp = world_down_prev = cam.v_world_up;
        if (cam.v_back != forward_prev) back_tmp = forward_prev = cam.v_back;

        ImGui::InputFloat3("center", glm::value_ptr(cam.center));
        ImGui::InputFloat3("origin", glm::value_ptr(cam.origin));
        static bool lock_fx_fy = true;
        ImGui::Checkbox("fx=fy", &lock_fx_fy);
        if (lock_fx_fy) {
            if (ImGui::SliderFloat("focal", &cam.fx, 300.f, 7000.f)) {
                cam.fy = cam.fx;
            }
        } else {
            ImGui::SliderFloat("fx", &cam.fx, 300.f, 7000.f);
            ImGui::SliderFloat("fy", &cam.fy, 300.f, 7000.f);
        }
        if (ImGui::TreeNode("Directions")) {
            ImGui::InputFloat3("world_up", glm::value_ptr(world_up_tmp));
            ImGui::InputFloat3("back", glm::value_ptr(back_tmp));
            if (ImGui::Button("normalize & update dirs")) {
                cam.v_world_up = glm::normalize(world_up_tmp);
                cam.v_back = glm::normalize(back_tmp);
            }
            ImGui::TreePop();
        }
    }  // End camera node

    ImGui::SetNextTreeNodeOpen(true, ImGuiCond_Once);
    if (ImGui::CollapsingHeader("Render")) {
        static float inv_step_size = 1.0f / rend.options.step_size;
        if (ImGui::SliderFloat("1/eps", &inv_step_size, 128.f, 20000.f)) {
            rend.options.step_size = 1.f / inv_step_size;
        }
        ImGui::SliderFloat("sigma_thresh", &rend.options.sigma_thresh, 0.f,
                           100.0f);
        ImGui::SliderFloat("stop_thresh", &rend.options.stop_thresh, 0.001f,
                           0.4f);
        ImGui::SliderFloat("bg_brightness", &rend.options.background_brightness,
                           0.f, 1.0f);

    }  // End render node
    ImGui::SetNextTreeNodeOpen(false, ImGuiCond_Once);
    if (ImGui::CollapsingHeader("Visualization")) {
        ImGui::PushItemWidth(230);
        ImGui::SliderFloat3("bb_min", rend.options.render_bbox, 0.0, 1.0);
        ImGui::SliderFloat3("bb_max", rend.options.render_bbox + 3, 0.0, 1.0);
        ImGui::SliderInt2("decomp", rend.options.basis_minmax, 0,
                          std::max(tree.data_format.basis_dim - 1, 0));
        ImGui::SliderFloat3("viewdir shift", rend.options.rot_dirs, -M_PI / 4,
                            M_PI / 4);
        ImGui::PopItemWidth();
        if (ImGui::Button("Reset Viewdir Shift")) {
            for (int i = 0; i < 3; ++i) rend.options.rot_dirs[i] = 0.f;
        }

#ifdef VOLREND_CUDA
        ImGui::Checkbox("Show Grid", &rend.options.show_grid);
        ImGui::SameLine();
        ImGui::Checkbox("Render Depth", &rend.options.render_depth);
        if (rend.options.show_grid) {
            ImGui::SliderInt("grid max depth", &rend.options.grid_max_depth, 0,
                             7);
        }
#endif
    }

#ifdef VOLREND_CUDA
    static bool animating = false;
    static std::string anim_path;
    static int anim_frame = 0;
    static cnpy::NpyArray anim_arr;

    static std::chrono::high_resolution_clock::time_point anim_tp;
    static double anim_last_seconds = 0.f;
    static const double anim_interval = 1.0 / 20;

    auto goto_frame = [&] {
        if (anim_path.empty()) return;

        if (anim_arr.word_size == 8) {
            const double* ptr =
                anim_arr.data<double>() + anim_frame * tree.n_joints * 3;
            for (int i = 0; i < tree.n_joints; ++i)
                for (int j = 0; j < 3; ++j) tree.pose[i][j] = ptr[i * 3 + j];
        } else {
            const float* ptr =
                anim_arr.data<float>() + anim_frame * tree.n_joints * 3;
            for (int i = 0; i < tree.n_joints; ++i)
                for (int j = 0; j < 3; ++j) tree.pose[i][j] = ptr[i * 3 + j];
        }
        tree.update_kintree();
    };
    if (animating) {
        auto now = std::chrono::high_resolution_clock::now();
        double seconds = std::chrono::duration<double>(now - anim_tp).count();
        if (seconds >= anim_last_seconds + anim_interval) {
            if (anim_frame < anim_arr.shape[0] - 1) {
                anim_frame++;
                goto_frame();
                anim_last_seconds += anim_interval;
            } else {
                animating = false;
            }
        }
    }

    open_animation_dialog.Display();
    if (open_animation_dialog.HasSelected()) {
        // Open animation
        anim_path = open_animation_dialog.GetSelected().string();
        std::cout << "Loading animation: " << anim_path << "\n";
        open_animation_dialog.ClearSelected();
        anim_arr = cnpy::npy_load(anim_path);
        if (anim_arr.shape.size() != 3 || anim_arr.shape[1] != tree.n_joints ||
            anim_arr.shape[2] != 3 ||
            (anim_arr.word_size != 8 && anim_arr.word_size != 4)) {
            std::cerr << "ERROR: invalid animation file\n";
            std::exit(1);
        }
        anim_frame = 0;
        goto_frame();
    }

    if (tree.is_rigged()) {
        ImGui::SetNextTreeNodeOpen(true, ImGuiCond_Once);
        if (ImGui::CollapsingHeader("Rigging")) {
            if (ImGui::Button("reset##rig-reset")) {
                tree.trans = glm::vec3(0);
                std::fill(tree.pose.begin(), tree.pose.end(), glm::vec3(0));
                tree.update_kintree();
            }
            ImGui::SameLine();
            if (ImGui::Button("deselect##rig-desel")) {
                rend.options.selected_joint = -2;
            }

            ImGui::SameLine();
            if (ImGui::Button("load anim##rig-loadanim")) {
                open_animation_dialog.Open();
            }
            static std::string sel_joint_name = "";
            if (rend.options.selected_joint >= 0) {
                sel_joint_name = joint_names[rend.options.selected_joint];
            } else if (rend.options.selected_joint == -2) {
                sel_joint_name = "";
            }
            ImGui::Checkbox("show joints", &rend.options.show_joints);
            if (rend.options.show_joints) {
                for (int j = 0; j < tree.n_joints; ++j) {
                    ImGuizmo::DrawCubes(glm::value_ptr(w2c),
                                        glm::value_ptr(camera_persp_prj),
                                        glm::value_ptr(tree.pose_mats[j]), 1);
                }
            }

            if (sel_joint_name.size()) {
                ImGui::Text("selected joint: %s", sel_joint_name.c_str());
            } else {
                ImGui::TextUnformatted("left click to select joint");
            }

            if (anim_path.size()) {
                ImGui::SetNextTreeNodeOpen(true, ImGuiCond_Once);
                if (ImGui::TreeNode("animation")) {
                    ImGui::TextUnformatted(anim_path.c_str());
                    if (ImGui::SliderInt("frame", &anim_frame, 0,
                                         anim_arr.shape[0])) {
                        goto_frame();
                    }
                    if (!animating) {
                        if (ImGui::Button("Play")) {
                            anim_last_seconds = 0.f;
                            anim_tp = std::chrono::high_resolution_clock::now();
                            animating = true;
                        }
                    } else {
                        if (ImGui::Button("Pause")) {
                            animating = false;
                        }
                    }
                    ImGui::TreePop();
                }
            }

            bool manip_updated = false;

            if (ImGui::TreeNode("trans")) {
                if (ImGui::SliderFloat3(
                        "trans##rig", glm::value_ptr(tree.trans), -1.f, 1.f)) {
                    manip_updated = true;
                }
                static glm::mat4 trans = glm::mat4(1.f);
                trans = glm::translate(glm::mat4(1.f), tree.trans);
                if (ImGuizmo::Manipulate(
                        glm::value_ptr(w2c), glm::value_ptr(camera_persp_prj),
                        ImGuizmo::TRANSLATE, ImGuizmo::LOCAL,
                        glm::value_ptr(trans), NULL, NULL, NULL, NULL)) {
                    tree.trans = glm::vec3(trans[3]);
                    manip_updated = true;
                }
                ImGui::TreePop();
            }
            const int STEP = 10;
            for (int j = 0; j < tree.n_joints; j += STEP) {
                int end_idx = std::min(j + STEP, tree.n_joints);
                std::string all_joint_names;
                for (int i = j; i < std::min(j + 2, tree.n_joints); ++i) {
                    const std::string& joint_name = joint_names[i];
                    if (i > j) all_joint_names.push_back(' ');
                    for (char c : joint_name) {
                        if (c != '_') all_joint_names.push_back(c);
                    }
                }
                all_joint_names.append("..");
                if (rend.options.selected_joint >= j &&
                    rend.options.selected_joint < end_idx) {
                    ImGui::SetNextTreeNodeOpen(true);
                }
                if (ImGui::TreeNode((std::to_string(j) + "-" +
                                     std::to_string(end_idx - 1) + ": " +
                                     all_joint_names)
                                        .c_str())) {
                    for (int i = j; i < end_idx; ++i) {
                        const std::string id = std::to_string(i);
                        std::string joint_name = joint_names[i];
                        joint_name += "##rig_" + id;
                        if (rend.options.selected_joint == i) {
                            ImGui::SetNextTreeNodeOpen(true);
                        } else if (~rend.options.selected_joint) {
                            ImGui::SetNextTreeNodeOpen(false);
                        }
                        if (ImGui::TreeNode(joint_name.c_str())) {
                            std::string slider_id = "axisangle##rig_sli_" + id;
                            if (ImGui::SliderFloat3(
                                    slider_id.c_str(),
                                    glm::value_ptr(tree.pose[i]), -M_PI / 2,
                                    M_PI / 2)) {
                                manip_updated = true;
                            }

                            ImGuizmo::SetID(i);
                            if (ImGuizmo::Manipulate(
                                    glm::value_ptr(w2c),
                                    glm::value_ptr(camera_persp_prj),
                                    ImGuizmo::ROTATE, ImGuizmo::LOCAL,
                                    glm::value_ptr(tree.pose_mats[i]), NULL,
                                    NULL, NULL, NULL, joint_names[i].c_str())) {
                                glm::mat3 rot = glm::mat3(tree.pose_mats[i]);
                                if (i) {
                                    rot = glm::transpose(glm::mat3(
                                              tree.pose_mats
                                                  [tree.kintree_table_[i]])) *
                                          rot;
                                }
                                glm::quat rot_q = glm::quat_cast(rot);
                                tree.pose[i] =
                                    glm::axis(rot_q) * glm::angle(rot_q);
                                manip_updated = true;
                            }
                            ImGui::TreePop();
                        }
                    }  // for i
                    ImGui::TreePop();
                }  // TreeNode Axis-angle
            }      // for j
            if (manip_updated) {
                tree.update_kintree();
            }
        }  // CollapsingHeader Rigging
        rend.options.selected_joint = -1;
    }  // if tree.is_rigged
    ImGui::SetNextTreeNodeOpen(false, ImGuiCond_Once);
    if (ImGui::CollapsingHeader("Manipulation")) {
        ImGui::BeginGroup();
        for (int i = 0; i < (int)rend.meshes.size(); ++i) {
            auto& mesh = rend.meshes[i];
            if (mesh.name.size() && mesh.name[0] != '_') {
                if (ImGui::TreeNode(mesh.name.c_str())) {
                    ImGui::PushItemWidth(230);
                    ImGui::SliderFloat3(
                        "trans", glm::value_ptr(mesh.translation), -2.0f, 2.0f);
                    ImGui::SliderFloat3("rot", glm::value_ptr(mesh.rotation),
                                        -M_PI, M_PI);
                    ImGui::SliderFloat("scale", &mesh.scale, 0.01f, 10.0f);
                    ImGui::PopItemWidth();
                    ImGui::Checkbox("visible", &mesh.visible);
                    ImGui::SameLine();
                    ImGui::Checkbox("unlit", &mesh.unlit);

                    ImGui::TreePop();
                }
            }
        }
        ImGui::EndGroup();
        if (ImGui::Button("Sphere")) {
            static int sphereid = 0;
            {
                Mesh sph = Mesh::Sphere();
                sph.scale = 0.1f;
                sph.translation[2] = 1.0f;
                sph.update();
                if (sphereid) sph.name = sph.name + std::to_string(sphereid);
                ++sphereid;
                rend.meshes.push_back(std::move(sph));
            }
        }
        ImGui::SameLine();
        if (ImGui::Button("Cube")) {
            static int cubeid = 0;
            {
                Mesh cube = Mesh::Cube();
                cube.scale = 0.2f;
                cube.translation[2] = 1.0f;
                cube.update();
                if (cubeid) cube.name = cube.name + std::to_string(cubeid);
                ++cubeid;
                rend.meshes.push_back(std::move(cube));
            }
        }
        ImGui::SameLine();
        if (ImGui::Button("Latti")) {
            static int lattid = 0;
            {
                Mesh latt = Mesh::Lattice();
                if (tree.N > 0) {
                    latt.scale =
                        1.f / std::min(std::min(tree.scale[0], tree.scale[1]),
                                       tree.scale[2]);
                    for (int i = 0; i < 3; ++i) {
                        latt.translation[i] =
                            -1.f / tree.scale[0] * tree.offset[0];
                    }
                }
                latt.update();
                if (lattid) latt.name = latt.name + std::to_string(lattid);
                ++lattid;
                rend.meshes.push_back(std::move(latt));
            }
        }
        ImGui::SameLine();
        if (ImGui::Button("Load Tri OBJ")) {
            open_obj_mesh_dialog.Open();
        }
        ImGui::SameLine();
        if (ImGui::Button("Clear All")) {
            rend.meshes.clear();
        }

        ImGui::BeginGroup();
        ImGui::Checkbox("Enable Lumisphere Probe", &rend.options.enable_probe);
        if (rend.options.enable_probe) {
            ImGui::SliderFloat3("probe", rend.options.probe, -2.f, 2.f);
            ImGui::SliderInt("probe_win_sz", &rend.options.probe_disp_size, 50,
                             800);
        }
        ImGui::EndGroup();
    }
    open_obj_mesh_dialog.Display();
    if (open_obj_mesh_dialog.HasSelected()) {
        // Load mesh
        auto sels = open_obj_mesh_dialog.GetMultiSelected();
        for (auto& fpath : sels) {
            const std::string path = fpath.string();
            Mesh tmp;
            std::cout << "Load OBJ: " << path << "\n";
            tmp.load_basic_obj(path);
            if (tmp.vert.size()) {
                // Auto offset
                std::ifstream ifs(path + ".offs");
                if (ifs) {
                    ifs >> tmp.translation.x >> tmp.translation.y >>
                        tmp.translation.z;
                    if (ifs) {
                        ifs >> tmp.scale;
                    }
                }
                tmp.update();
                rend.meshes.push_back(std::move(tmp));
                std::cout << "Load success\n";
            } else {
                std::cout << "Load failed\n";
            }
        }
        open_obj_mesh_dialog.ClearSelected();
    }

#endif
    ImGui::End();

    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

}  // namespace
}  // namespace volrend

int main(int argc, char* argv[]) {
    using namespace volrend;
    cxxopts::Options cxxoptions(
        "volrend",
        "OpenGL PlenOctree volume rendering (c) PlenOctree authors 2021");

    internal::add_common_opts(cxxoptions);
    // clang-format off
    cxxoptions.add_options()
        ("nogui", "disable imgui", cxxopts::value<bool>())
        ("joints", "joint names file", cxxopts::value<std::string>()->default_value(""))
        ("center", "camera center position (world); ignored for NDC",
                cxxopts::value<std::vector<float>>()->default_value(
                                                        "-3.5,0,4.5"))
        ("back", "camera's back direction unit vector (world) for orientation; ignored for NDC",
                cxxopts::value<std::vector<float>>()->default_value("-0.7071068,0,0.7071068"))
        ("origin", "origin for right click rotation controls; ignored for NDC",
                cxxopts::value<std::vector<float>>()->default_value("0,0,1"))
        ("world_up", "world up direction for rotating controls e.g. "
                     "0,0,1=blender; ignored for NDC",
                cxxopts::value<std::vector<float>>()->default_value("0,0,1"))
        ("grid", "show grid with given max resolution (4 is reasonable)", cxxopts::value<int>())
        ("probe", "enable lumisphere_probe and place it at given x,y,z",
                   cxxopts::value<std::vector<float>>())
        ;
    // clang-format on

    cxxoptions.positional_help("npz_file");

    cxxopts::ParseResult args = internal::parse_options(cxxoptions, argc, argv);

#ifdef VOLREND_CUDA
    const int device_id = args["gpu"].as<int>();
    if (~device_id) {
        cuda(SetDevice(device_id));
    }
#endif

    N3Tree tree;
    bool init_loaded = false;
    if (args.count("file")) {
        init_loaded = true;
        tree.open(args["file"].as<std::string>(),
                  args["rig"].as<std::string>());
    }
    int width = args["width"].as<int>(), height = args["height"].as<int>();
    float fx = args["fx"].as<float>();
    float fy = args["fy"].as<float>();
    bool nogui = args["nogui"].as<bool>();

    std::vector<std::string> joint_names;
    {
        joint_names.reserve(tree.n_joints);
        const std::string joint_name_file = args["joints"].as<std::string>();
        if (joint_name_file.size()) {
            std::ifstream ifs(joint_name_file);
            std::string joint_name;
            while (ifs >> joint_name) {
                joint_names.push_back(joint_name);
            }
        }
        for (int i = joint_names.size(); i < tree.n_joints; ++i) {
            joint_names.push_back("_JOINT_" + std::to_string(i));
        }
    }

    GLFWwindow* window = glfw_init(width, height);

    {
        VolumeRenderer rend;
        if (fx > 0.f) {
            rend.camera.fx = fx;
        }

        rend.options = internal::render_options_from_args(args);
        if (init_loaded && tree.use_ndc) {
            // Special inital coordinates for NDC
            // (pick average camera)
            rend.camera.center = glm::vec3(0);
            rend.camera.origin = glm::vec3(0, 0, -3);
            rend.camera.v_back = glm::vec3(0, 0, 1);
            rend.camera.v_world_up = glm::vec3(0, 1, 0);
            if (fx <= 0) {
                rend.camera.fx = rend.camera.fy = tree.ndc_focal * 0.25f;
            }
            rend.camera.movement_speed = 0.1f;
        } else {
            auto cen = args["center"].as<std::vector<float>>();
            rend.camera.center = glm::vec3(cen[0], cen[1], cen[2]);
            auto origin = args["origin"].as<std::vector<float>>();
            rend.camera.origin = glm::vec3(origin[0], origin[1], origin[2]);
            auto world_up = args["world_up"].as<std::vector<float>>();
            rend.camera.v_world_up =
                glm::vec3(world_up[0], world_up[1], world_up[2]);
            auto back = args["back"].as<std::vector<float>>();
            rend.camera.v_back = glm::vec3(back[0], back[1], back[2]);
        }
        if (fy <= 0.f) {
            rend.camera.fy = rend.camera.fx;
        }
        glfwGetFramebufferSize(window, &width, &height);
        rend.set(tree);
        rend.resize(width, height);

        // std::vector<size_t> joint_mesh_ids(tree.n_joints);
        // for (int i = 0; i < tree.n_joints; ++i) {
        //     Mesh sphere = Mesh::Sphere();
        //     sphere.name = "_joint_#" + std::to_string(i);
        //     sphere.translation = tree.joint_pos_posed_[i];
        //     sphere.scale = 0.02f;
        //     sphere.update();
        //     joint_mesh_ids[i] = rend.meshes.size();
        //     rend.meshes.push_back(std::move(sphere));
        // }

        // Set user pointer and callbacks
        glfwSetWindowUserPointer(window, &rend);
        glfwSetKeyCallback(window, glfw_key_callback);
        glfwSetMouseButtonCallback(window, glfw_mouse_button_callback);
        glfwSetCursorPosCallback(window, glfw_cursor_pos_callback);
        glfwSetScrollCallback(window, glfw_scroll_callback);
        glfwSetFramebufferSizeCallback(window, glfw_window_size_callback);

        while (!glfwWindowShouldClose(window)) {
#ifdef VOLREND_CUDA
            glEnable(GL_DEPTH_TEST);
#endif
            glEnable(GL_PROGRAM_POINT_SIZE);
            glPointSize(4.f);
            glfw_update_title(window);

            // for (int i = 0; i < tree.n_joints; ++i) {
            //     auto& sphere = rend.meshes[i];
            //     sphere.translation = tree.joint_pos_posed_[i];
            // }

            rend.render();

            if (!nogui) draw_imgui(rend, tree, joint_names);

            glfwSwapBuffers(window);
            glFinish();
            glfwPollEvents();
        }
    }

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(window);
    glfwTerminate();
}
