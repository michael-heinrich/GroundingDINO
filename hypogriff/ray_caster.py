import numpy as np

class RayCaster:
    '''
    Allows to convert between image coordinates and rays in world coordinates.
    Expects the matrices to be in column-major order.
    '''
    def __init__(self, matrix_world: np.ndarray, projection_matrix_inverse: np.ndarray, screeWidth: int, screenHeight: int):
        self.matrix_world = matrix_world
        self.projection_matrix_inverse = projection_matrix_inverse
        self.screenWidth = screeWidth
        self.screenHeight = screenHeight


    def apply_matrix4(self, matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
        '''
        Applies a 4x4 matrix to a 3D vector. (Exactly the same as in three.js)
        Expecting the matrix to be in column-major order.
        '''
        
        x = vector[0]
        y = vector[1]
        z = vector[2]
        e = matrix

        w = 1.0 / (e[3] * x + e[7] * y + e[11] * z + e[15])

        return np.array([
            (e[0] * x + e[4] * y + e[8] * z + e[12]) * w,
            (e[1] * x + e[5] * y + e[9] * z + e[13]) * w,
            (e[2] * x + e[6] * y + e[10] * z + e[14]) * w
        ])
    
    def get_matrix_position(self, matrix: np.ndarray) -> np.ndarray:
        '''
        Returns the position vector of a 4x4 matrix.
        Expecting the matrix to be in column-major order.
        '''
        return np.array([matrix[12], matrix[13], matrix[14]])
    

    def normalized_device_coords_to_ray(self, x: float, y: float) -> np.ndarray:
        '''
        Converts normalized device coordinates to a direction vector in world coordinates.
        Exactly the same semantics as Raycaster.setFromCamera() in three.js.
        '''

        origin = self.get_matrix_position(self.matrix_world)

        # set z to 0.5 to get a direction vector
        normalized_dev = np.array([x, y, 0.5])

        # apply inverse projection matrix
        tmp1 = self.apply_matrix4(self.projection_matrix_inverse, normalized_dev)

        # apply matrix_world
        tmp2 = self.apply_matrix4(self.matrix_world, tmp1)

        # subtract origin
        dir = tmp2 - origin

        # normalize
        normalized_dir = dir / np.linalg.norm(dir)

        return (origin, normalized_dir)


    def image_coords_to_ray(self, x: float, y: float) -> np.ndarray:
        '''
        Converts image coordinates to a direction vector in world coordinates.
        '''
        # convert to normalized device coordinates
        normalized_dev_x = (x / self.screenWidth) * 2 - 1
        normalized_dev_y = -(y / self.screenHeight) * 2 + 1

        return self.normalized_device_coords_to_ray(normalized_dev_x, normalized_dev_y)
        
