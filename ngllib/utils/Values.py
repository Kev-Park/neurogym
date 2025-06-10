import numpy as np

class Values:
    def __init__(self):

        # DELTA JSON STATE - Actor Critic
        self.delta_x_factor = 5000  # delta_position_x
        self.delta_y_factor = 5000  # delta_position_y
        self.delta_z_factor = 10 # delta_position_z
        self.delta_crossSectionScale_factor = 5  # delta_crossSectionScale
        self.delta_q1_factor = 1  # delta_projectionOrientation_q1
        self.delta_q2_factor = 1  # delta_projectionOrientation_q2
        self.delta_q3_factor = 1  # delta_projectionOrientation_q3
        self.delta_q4_factor = 1  # delta_projectionOrientation_q4
        self.delta_projectionScale_factor = 1  # delta_projectionScale

        # FOR THE POSITION STATE NORMALIZATION - based on the 1300 episodes from Fast-IL
        self.pos_x_mean = 157510
        self.pos_x_std = 278.76
        self.pos_y_mean = 42018
        self.pos_y_std = 286
        self.pos_z_mean = 4661
        self.pos_z_std = 160.76

        self.crossSectionScale_mean = 0.2605397078599756
        self.crossSectionScale_std = max(1.1102230246251565e-16 + 1e-6,1)

        self.euler_angles_x_mean = 0
        self.euler_angles_x_std = np.sqrt(np.pi)
        self.euler_angles_y_mean = 0
        self.euler_angles_y_std = np.sqrt(np.pi)
        self.euler_angles_z_mean = 0
        self.euler_angles_z_std = np.sqrt(np.pi)

        self.projectionScale_mean = 1890
        self.projectionScale_std = 113

        # self.click_pos_x_mean = 1338.51
        # self.click_pos_x_std = 360.4778244219747
        # self.click_pos_y_mean = 478.352
        # self.click_pos_y_std = 323.13848439330155

        # self.z_projected_x_mean = -0.04214604056155423
        # self.z_projected_x_std = 0.7985694889066879
        # self.z_projected_y_mean = 0.004619252266395272
        # self.z_projected_y_std = 0.6004074825989384

        # From the FLywire dataset, we have the following values
        self.min_position= [20400, 5760, 16]
        self.max_position= [236800, 118400, 7062]
        self.min_crossSectionScale= 1
        self.max_crossSectionScale= 300
        self.max_projectionOrientation= [1, 1, 1, 1]
        self.min_projectionScale= 1
        self.max_projectionScale= 25000 # this is a very high value # previously 1000000

        self.position_x_factor =  self.max_position[0]
        self.position_y_factor =  self.max_position[1]
        self.position_z_factor =  self.max_position[2]
        self.crossSectionScale_factor = self.max_crossSectionScale
        self.projectionOrientation_q1_factor = self.max_projectionOrientation[0]
        self.projectionOrientation_q2_factor = self.max_projectionOrientation[1]
        self.projectionOrientation_q3_factor = self.max_projectionOrientation[2]
        self.projectionOrientation_q4_factor = self.max_projectionOrientation[3]

        self.projectionOrientation_euler1_factor = 3.141592653589793
        self.projectionOrientation_euler2_factor = 3.141592653589793
        self.projectionOrientation_euler3_factor = 3.141592653589793

        self.projectionScale_factor = self.max_projectionScale


        self.data_image_width = 1800
        self.data_image_height = 900
        self.neuroglancer_margin_top = 20
        self.model_image_width = 960
        self.model_image_height = 540

        # ABSOLUTE MOUSE POSITION
        self.x_factor = self.data_image_width
        self.y_factor = self.data_image_height
        # PROGRAMMATIC MOUSE POSITION
        self.x_factor_programmatic = 1
        self.y_factor_programmatic = 1

        # Discrete actions are in order: left_click, right_click, double_click, Shift, Ctrl, Alt, JSON_change
        # Continuous actions are in order: x, y, delta_position_x, delta_position_y, delta_position_z, delta_crossSectionScale, delta_projectionOrientation_q1, q2, q3, q4, delta_projectionScale
        # Here represents the actions and their indexes in the output vector
        self.discrete_action_space_indexes = {"Left click": 0, 
                                              "Right click": 1,
                                              "Double click": 2, 
                                              "Shift": 3,
                                              "Ctrl": 4,
                                              "Alt": 5,
                                              "JSON change": 6}
        self.discrete_action_number = 7
        self.continuous_action_space_indexes = {"x": 0,
                                                "y": 1,
                                                "delta_position_x": 2,
                                                "delta_position_y": 3,
                                                "delta_position_z": 4,
                                                "delta_crossSectionScale": 5,
                                                "delta_projectionOrientation_q1": 6,
                                                "delta_projectionOrientation_q2": 7,
                                                "delta_projectionOrientation_q3": 8,
                                                "delta_projectionOrientation_q4": 9,
                                                "delta_projectionScale": 10}
        self.continuous_action_number = 11
        self.AC_reward_factor = 0.001


        self.grid_size_x = 25
        self.grid_size_y = 25 
        viewport_x_size = self.data_image_width
        viewport_y_size = self.data_image_height # not taking into account the neuroglancer margin, the model sees the whole image, we take care after of margin issues in clicking
        self.number_of_x_cells = viewport_x_size // self.grid_size_x
        self.number_of_y_cells = viewport_y_size // self.grid_size_y


        
        self.Q_action_space_indexes = {
                                        # Click actions
                                        "Left click": 0,
                                        "Right click": 1,
                                        "Double click": 2,

                                        # Mouse movement actions - X
                                        "incr_mouse_x_500": 3,
                                        "incr_mouse_x_100": 4,
                                        "incr_mouse_x_50": 5,
                                        "incr_mouse_x_10": 6,
                                        "incr_mouse_x_5": 7,
                                        "incr_mouse_x_1": 8,
                                        "decr_mouse_x_500": 9,
                                        "decr_mouse_x_100": 10,
                                        "decr_mouse_x_50": 11,
                                        "decr_mouse_x_10": 12,
                                        "decr_mouse_x_5": 13,
                                        "decr_mouse_x_1": 14,

                                        # Mouse movement actions - Y
                                        "incr_mouse_y_500": 15,
                                        "incr_mouse_y_100": 16,
                                        "incr_mouse_y_50": 17,
                                        "incr_mouse_y_10": 18,
                                        "incr_mouse_y_5": 19,
                                        "incr_mouse_y_1": 20,
                                        "decr_mouse_y_500": 21,
                                        "decr_mouse_y_100": 22,
                                        "decr_mouse_y_50": 23,
                                        "decr_mouse_y_10": 24,
                                        "decr_mouse_y_5": 25,
                                        "decr_mouse_y_1": 26,

                                        # Position actions - X
                                        "incr_position_x_200": 27,
                                        "incr_position_x_50": 28,
                                        "incr_position_x_5": 29,
                                        "incr_position_x_1": 30,
                                        "decr_position_x_200": 31,
                                        "decr_position_x_50": 32,
                                        "decr_position_x_5": 33,
                                        "decr_position_x_1": 34,

                                        # Position actions - Y
                                        "incr_position_y_200": 35,
                                        "incr_position_y_50": 36,
                                        "incr_position_y_5": 37,
                                        "incr_position_y_1": 38,
                                        "decr_position_y_200": 39,
                                        "decr_position_y_50": 40,
                                        "decr_position_y_5": 41,
                                        "decr_position_y_1": 42,

                                        # Position actions - Z
                                        "incr_position_z_200": 43,
                                        "incr_position_z_50": 44,
                                        "incr_position_z_5": 45,
                                        "incr_position_z_1": 46,
                                        "decr_position_z_200": 47,
                                        "decr_position_z_50": 48,
                                        "decr_position_z_5": 49,
                                        "decr_position_z_1": 50,

                                        # CrossSectionScale actions
                                        "incr_crossSectionScale_1": 51,
                                        "incr_crossSectionScale_0.1": 52,
                                        "decr_crossSectionScale_1": 53,
                                        "decr_crossSectionScale_0.1": 54,

                                        # ProjectionOrientation actions
                                        "incr_projectionOrientation_q1_0.3": 55,
                                        "incr_projectionOrientation_q1_0.1": 56,
                                        "decr_projectionOrientation_q1_0.3": 57,
                                        "decr_projectionOrientation_q1_0.1": 58,
                                        "incr_projectionOrientation_q2_0.3": 59,
                                        "incr_projectionOrientation_q2_0.1": 60,
                                        "decr_projectionOrientation_q2_0.3": 61,
                                        "decr_projectionOrientation_q2_0.1": 62,
                                        "incr_projectionOrientation_q3_0.3": 63,
                                        "incr_projectionOrientation_q3_0.1": 64,
                                        "decr_projectionOrientation_q3_0.3": 65,
                                        "decr_projectionOrientation_q3_0.1": 66,

                                        # ProjectionScale actions
                                        "incr_projectionScale_5000": 67,
                                        "incr_projectionScale_1000": 68,
                                        "incr_projectionScale_100": 69,
                                        "decr_projectionScale_5000": 70,
                                        "decr_projectionScale_1000": 71,
                                        "decr_projectionScale_100": 72
                                    }
        
        self.click_limit_index = self.Q_action_space_indexes["decr_mouse_y_1"] #dangerous to hard code like this. TODO:...fix-it

        self.Q_action_translation_dict = {index: action 
                        for action, index in self.Q_action_space_indexes.items()}
        
        self.Q_action_space_indexes_grid = {
                                        # Click actions
                                        "Left click": 0,
                                        "Right click": 1,
                                        "Double click": 2,

                                        # Position actions - X
                                        "incr_position_x_1000": 3,
                                        "incr_position_x_100": 4,
                                        "incr_position_x_5": 5,
                                        "incr_position_x_1": 6,
                                        "decr_position_x_1000": 7,
                                        "decr_position_x_100": 8,
                                        "decr_position_x_5": 9,
                                        "decr_position_x_1": 10,

                                        # Position actions - Y
                                        "incr_position_y_1000": 11,
                                        "incr_position_y_100": 12,
                                        "incr_position_y_5": 13,
                                        "incr_position_y_1": 14,
                                        "decr_position_y_1000": 15,
                                        "decr_position_y_100": 16,
                                        "decr_position_y_5": 17,
                                        "decr_position_y_1": 18,

                                        # Position actions - Z
                                        "incr_position_z_1000": 19,
                                        "incr_position_z_100": 20,
                                        "incr_position_z_5": 21,
                                        "incr_position_z_1": 22,
                                        "decr_position_z_1000": 23,
                                        "decr_position_z_100": 24,
                                        "decr_position_z_5": 25,
                                        "decr_position_z_1": 26,

                                        # CrossSectionScale actions
                                        "incr_crossSectionScale_1": 27,
                                        "incr_crossSectionScale_0.1": 28,
                                        "decr_crossSectionScale_1": 29,
                                        "decr_crossSectionScale_0.1": 30,

                                        # ProjectionOrientation actions -- EULER ANGLES # previously quaternions
                                        "incr_projectionOrientation_q1_1.0": 31,
                                        "incr_projectionOrientation_q1_0.2": 32,
                                        "decr_projectionOrientation_q1_1.0": 33,
                                        "decr_projectionOrientation_q1_0.2": 34,
                                        "incr_projectionOrientation_q2_1.0": 35,
                                        "incr_projectionOrientation_q2_0.2": 36,
                                        "decr_projectionOrientation_q2_1.0": 37,
                                        "decr_projectionOrientation_q2_0.2": 38,
                                        "incr_projectionOrientation_q3_1.0": 39,
                                        "incr_projectionOrientation_q3_0.2": 40,
                                        "decr_projectionOrientation_q3_1.0": 41,
                                        "decr_projectionOrientation_q3_0.2": 42,

                                        # ProjectionScale actions
                                        "incr_projectionScale_5000": 43,
                                        "incr_projectionScale_1000": 44,
                                        "incr_projectionScale_100": 45,
                                        "decr_projectionScale_5000": 46,
                                        "decr_projectionScale_1000": 47,
                                        "decr_projectionScale_100": 48,
                                    }
        self.click_limit_index_grid = self.Q_action_space_indexes_grid["Double click"] #dangerous to hard code like this. TODO:...fix-it
        # Adding dynamically generated actions for move_to_box_X_Y
        grid_offset = len(self.Q_action_space_indexes_grid)  # Start after defined actions

        self.Q_action_space_indexes_grid.update({
            f"move_to_box_{x * self.grid_size_x}_{y * self.grid_size_y}": grid_offset + x * self.number_of_y_cells + y
            for x in range(self.number_of_x_cells)
            for y in range(self.number_of_y_cells)
        })

        self.grid_only_actions = {}
        self.grid_only_actions.update({
            f"move_to_box_{x * self.grid_size_x}_{y * self.grid_size_y}": x * self.number_of_y_cells + y
            for x in range(self.number_of_x_cells)
            for y in range(self.number_of_y_cells)
        })
        self.num_Q_actions = len(self.Q_action_space_indexes)
        self.num_Q_actions_grid = len(self.Q_action_space_indexes_grid)
        self.num_grid_only_actions = len(self.grid_only_actions)

        self.Q_action_translation_dict_grid = {index: action 
                for action, index in self.Q_action_space_indexes_grid.items()}

        self.grid_action_translation_dict = {index: action
                for action, index in self.grid_only_actions.items()}
        
        self.q_reward_factor = 0.001 # previously 0.00001
        self.q_reward_delta_z_factor = 0.001

    def get_reward_from_pos_state(self, pos_state):
        # Pos state is not normalized here
        #print("Pos state", pos_state)
        z_position = pos_state[0][2]

        return z_position*self.q_reward_factor
    
    def get_reward_from_normalized_pos_state(self, norm_pos_state):
        # Pos state is normalized here
        #print("Pos state", pos_state)
        norm_z_position = norm_pos_state[0][2]

        return (norm_z_position*self.position_z_factor)*self.q_reward_factor
    
    def get_reward_from_pos_state_delta(self, pos_state_0, pos_state_1):
        # Delta pos state is not normalized here
        #print("Pos state", pos_state)
        z_position = pos_state_0[0][2]
        z_position_next = pos_state_1[0][2]
        delta = z_position_next - z_position
        return delta*self.q_reward_delta_z_factor
    
    def map_to_grid(self, mouse_x, mouse_y):
        #print("Mouse position", mouse_x, mouse_y)
        mouse_x = min(max(0, mouse_x), self.data_image_width)
        mouse_y = min(max(0, mouse_y), self.data_image_height)
        column = mouse_x // self.grid_size_x
        row = mouse_y // self.grid_size_y
        mouse_pos_x = column * self.grid_size_x
        mouse_pos_y = row * self.grid_size_y
        #print("Mouse position after", mouse_pos_x, mouse_pos_y)
        return (mouse_pos_x, mouse_pos_y)

    def click_to_action_index(self, click_pos, index_dictionary):
        mouse_x = click_pos[0]
        mouse_y = click_pos[1]
        r_mouse_x, r_mouse_y = self.map_to_grid(mouse_x, mouse_y)

        return index_dictionary[f"move_to_box_{r_mouse_x}_{r_mouse_y}"]
    