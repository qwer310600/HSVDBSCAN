from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import colorsys
import webcolors
import matplotlib.colors as mcolors
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN

from misc_utils import check, is_list_of_type, lookat_matrix
from scannet_utils import ObjInstance
from scope_env import GlobalState

SCORE_FUNCTIONS: dict[str, type[ScoreFuncBase]] = {}


def register_score_func(score_func_class: type[ScoreFuncBase]):
    assert hasattr(score_func_class, "NEED_ANCHOR")
    assert isinstance(score_func_class.NEED_ANCHOR, bool)

    assert hasattr(score_func_class, "NAME")

    if isinstance(score_func_class.NAME, str):
        assert score_func_class.NAME not in SCORE_FUNCTIONS
        SCORE_FUNCTIONS[score_func_class.NAME] = score_func_class

    elif is_list_of_type(score_func_class.NAME, str):
        for name in score_func_class.NAME:
            assert name not in SCORE_FUNCTIONS
            SCORE_FUNCTIONS[name] = score_func_class

    else:
        raise SystemError(f"invalid score_func NAME: {score_func_class.NAME}")


class ScoreFuncBase(ABC):
    NAME: str | None = None
    NEED_ANCHOR: bool | None = None

    @staticmethod
    @abstractmethod
    def get_scores(
        candidate_instances: list[ObjInstance],
        anchor: ObjInstance | None = None,
    ) -> list[float]:
        """compute the score for each instance. anchor is optional."""


@register_score_func
class ScoreDistance(ScoreFuncBase):
    NAME = "distance"
    NEED_ANCHOR = True

    @staticmethod
    def get_scores(
        candidate_instances: list[ObjInstance],
        anchor: ObjInstance | None = None,
    ) -> list[float]:
        return [
            np.linalg.norm(x.bbox.center - anchor.bbox.center)
            for x in candidate_instances
        ]


@register_score_func
class ScoreSizeX(ScoreFuncBase):
    NAME = "size-x"
    NEED_ANCHOR = False

    @staticmethod
    def get_scores(
        candidate_instances: list[ObjInstance],
        anchor: ObjInstance | None = None,
    ) -> list[float]:
        return [x.bbox.size[0] for x in candidate_instances]


@register_score_func
class ScoreSizeY(ScoreFuncBase):
    NAME = "size-y"
    NEED_ANCHOR = False

    @staticmethod
    def get_scores(
        candidate_instances: list[ObjInstance],
        anchor: ObjInstance | None = None,
    ) -> list[float]:
        return [x.bbox.size[1] for x in candidate_instances]


@register_score_func
class ScoreSizeZ(ScoreFuncBase):
    NAME = "size-z"
    NEED_ANCHOR = False

    @staticmethod
    def get_scores(
        candidate_instances: list[ObjInstance],
        anchor: ObjInstance | None = None,
    ) -> list[float]:
        return [x.bbox.size[2] for x in candidate_instances]


@register_score_func
class ScoreMaxSize(ScoreFuncBase):
    NAME = "size"
    NEED_ANCHOR = False

    @staticmethod
    def get_scores(
        candidate_instances: list[ObjInstance],
        anchor: ObjInstance | None = None,
    ) -> list[float]:
        return [x.bbox.max_extent for x in candidate_instances]


@register_score_func
class ScorePositionZ(ScoreFuncBase):
    NAME = "position-z"
    NEED_ANCHOR = False

    @staticmethod
    def get_scores(
        candidate_instances: list[ObjInstance],
        anchor: ObjInstance | None = None,
    ) -> list[float]:
        return [x.bbox.center[2] for x in candidate_instances]


@register_score_func
class ScoreLeft(ScoreFuncBase):
    NAME = "left"
    NEED_ANCHOR = False

    @staticmethod
    def get_scores(
        candidate_instances: list[ObjInstance],
        anchor: ObjInstance | None = None,
    ) -> list[float]:
        cand_center = np.mean([x.bbox.center for x in candidate_instances], axis=0)
        # look at the center of all candidate instances from the room center
        world_to_local = lookat_matrix(eye=cand_center, target=GlobalState.room_center)

        return [
            -(world_to_local @ np.hstack([x.bbox.center, 1]))[0]
            for x in candidate_instances
        ]


@register_score_func
class ScoreRight(ScoreFuncBase):
    NAME = "right"
    NEED_ANCHOR = False

    @staticmethod
    def get_scores(
        candidate_instances: list[ObjInstance],
        anchor: ObjInstance | None = None,
    ) -> list[float]:
        cand_center = np.mean([x.bbox.center for x in candidate_instances], axis=0)
        # look at the center of all candidate instances from the room center
        world_to_local = lookat_matrix(eye=cand_center, target=GlobalState.room_center)

        return [
            (world_to_local @ np.hstack([x.bbox.center, 1]))[0]
            for x in candidate_instances
        ]


@register_score_func
class ScoreFront(ScoreFuncBase):
    NAME = "front"
    NEED_ANCHOR = False

    @staticmethod
    def get_scores(
        candidate_instances: list[ObjInstance],
        anchor: ObjInstance | None = None,
    ) -> list[float]:
        cand_center = np.mean([x.bbox.center for x in candidate_instances], axis=0)
        # look at the center of all candidate instances from the room center
        world_to_local = lookat_matrix(eye=cand_center, target=GlobalState.room_center)

        # the larger the z-coord value, the nearer the instance is to the room center, i.e. "to the front"
        return [
            (world_to_local @ np.hstack([x.bbox.center, 1]))[2]
            for x in candidate_instances
        ]


@register_score_func
class ScoreCenter(ScoreFuncBase):
    NAME = ["distance-to-center", "distance-to-middle"]
    NEED_ANCHOR = False

    @staticmethod
    def get_scores(
        candidate_instances: list[ObjInstance],
        anchor: ObjInstance | None = None,
    ) -> list[float]:
        center = np.mean([x.bbox.center for x in candidate_instances], axis=0)
        return [np.linalg.norm(x.bbox.center - center) for x in candidate_instances]
    
COLOR_FAMILIES = {
    "red_family": ["snow", "rosybrown", "lightcoral", "indianred", "brown", "firebrick", "maroon", "darkred", "red", "mistyrose", 
                   "salmon", "tomato", "darksalmon", "coral", "orangered", "lightsalmon", "sienna", "seashell", "chocolate", "saddlebrown", 
                   "sandybrown", "peachpuff",  "peru", "linen", "bisque", "darkorange", "burlywood", "antiquewhite", "tan", "navajowhite", 
                   "blanchedalmond", "papayawhip", "moccasin", "orange", "wheat", "oldlace", "floralwhite", "darkgoldenrod", "goldenrod", "cornsilk",
                   "gold", "lemonchiffon", "khaki", "palegoldenrod", "darkkhaki", "ivory", "beige", "lightyellow", "lightgoldenrodyellow", "olive", 
                   "mediumorchid", "thistle", "plum", "violet", "purple", "darkmagenta", "fuchsia", "magenta", "orchid", "mediumvioletred", 
                   "deeppink", "hotpink", "lavenderblush", "palevioletred", "crimson", "pink", "lightpink"],
    "yellow_family": ["sandybrown", "peachpuff", "peru", "linen", "bisque", "darkorange", "burlywood", "antiquewhite", "tan", "navajowhite", 
                      "blanchedalmond", "papayawhip", "moccasin", "orange", "wheat", "oldlace", "floralwhite", "darkgoldenrod", "goldenrod", "cornsilk", 
                      "gold", "lemonchiffon", "khaki", "palegoldenrod", "darkkhaki", "ivory", "beige", "lightyellow", "lightgoldenrodyellow", "olive", "yellow", "brown"],
    "green_family": ["cornsilk", "gold", "lemonchiffon", "khaki", "palegoldenrod", "darkkhaki", "ivory", "beige", "lightyellow", "lightgoldenrodyellow", 
                     "olive", "yellow", "olivedrab", "yellowgreen", "darkolivegreen", "greenyellow", "chartreuse", "lawngreen", "honeydew", "darkseagreen",
                     "palegreen", "lightgreen", "forestgreen", "limegreen", "darkgreen", "green", "seagreen", "mediumseagreen", "springgreen", "mintcream" "mediumspringgreen", 
                     "mediumaquamarine", "aquamarine", "turquoise", "lightseagreen", "mediumturquoise", "azure", "lightcyan", "paleturquoise", "darkslategray", "darkslategrey", 
                     "teal", "darkcyan", "aqua", "cyan", "darkturquoise", "cadetblue", "powderblue"],
    "blue_family": ["mediumaquamarine", "aquamarine", "turquoise", "lightseagreen", "mediumturquoise", "azure", "lightcyan", "paleturquoise", "darkslategray", "darkslategrey", 
                    "teal", "darkcyan", "aqua", "cyan", "darkturquoise", "cadetblue", "powderblue", "lightblue", "deepskyblue", "skyblue", 
                    "lightskyblue", "steelblue", "aliceblue", "dodgerblue", "lightslategray", "lightslategrey", "slategray", "lightsteelblue", "cornflowerblue", "royalblue", 
                    "blue", "mediumblue", "darkblue", "midnightblue", "navy", "darkslateblue", "slateblue", "mediumslateblue", "mediumpurple", "rebeccapurple", 
                    "blueviolet", "indigo"],
    "purple_family": ["slateblue", "mediumslateblue", "mediumpurple", "rebeccapurple", "blueviolet", "indigo", "darkorchid", "darkviolet", "mediumorchid", "thistle", 
                      "plum", "violet", "purple", "darkmagenta", "fuchsia", "magenta", "orchid", "mediumvioletred", "deeppink", "hotpink", 
                      "lavenderblush", "palevioletred"],
    "black_family": ["black", "dimgray", "dimgrey", "gray", "grey", "darkgray", "darkgrey", "silver", "rosybrown", "maroon", 
                     "darkred", "sienna", "saddlebrown", "darkolivegreen", "darkseagreen", "darkgreen", "darkslategray", "darkslategrey", "midnightblue", "navy", 
                     "darkblue", "darkslateblue", "rebeccapurple", "indigo", "darkmagenta"],
    "white_family": ["silver", "lightgray", "lightgrey", "gainsboro", "whitesmoke", "white", "snow", "mistyrose", "seashell" "peachpuff", 
                     "linen", "bisque", "antiquewhite", "blanchedalmond", "papayawhip", "oldlace", "floralwhite", "cornsilk", "ivory", "beige", 
                     "lightyellow", "lightgoldenrodyellow", "honeydew", "mintcream", "azure", "lightcyan", "aliceblue", "lightsteelblue" "ghostwhite", "lavender", 
                     "thisle", "lavenderblush", "pink", "lightpink"]
}

"""
@register_score_func
class ScoreColor(ScoreFuncBase):
    NAME = "color"
    NEED_ANCHOR = False  # No longer needed

    @staticmethod
    def get_scores(
        candidate_instances: list['ObjInstance'],
        condition: str
    ) -> bool:

        # Assume there is always 1 candidate instance
        obj = candidate_instances[0]
        #print("obj: ", obj.inst_id)

        # Extract object's color
        if hasattr(obj, 'vertices') and obj.vertices.shape[1] >= 6:
            #alpha = obj.vertices[:, 6]  # Extract alpha values
            #avg_alpha = np.mean(alpha)  # Calculate the average alpha
            #print("avg_alpha: ", avg_alpha)
            colors = obj.vertices[:, 3:6]  # Extract RGB values
            avg_color = np.mean(colors, axis=0)  # Calculate the average color
            obj_color = tuple(map(int, avg_color * 255))  # Convert to (R, G, B)
        else:
            raise ValueError(f"Instance {obj.inst_id} does not have RGB values in vertices.")

        # Convert RGB to the closest color name
        # 1) Normalize RGB values to range 0-1
        rgb_normalized = tuple([x / 255.0 for x in obj_color])

        # 2) Find the closest color from the matplotlib color list
        closest_name = None
        min_diff = float('inf')  # Store the smallest color difference

        for name, hex_color in mcolors.CSS4_COLORS.items():
            # Convert the color name to RGB
            color_rgb = mcolors.hex2color(hex_color)
            # Calculate the RGB difference
            diff = sum((a - b) ** 2 for a, b in zip(rgb_normalized, color_rgb))
            if diff < min_diff:
                min_diff = diff
                closest_name = name

        #print(f"Closest color to object: {closest_name}")
        #print(f"Condition color: {condition}")

        # Function to get the color families for a given color
        def get_color_families(color_name):
            families = set() 
            for family, colors in COLOR_FAMILIES.items():
                if color_name in colors:
                    families.add(family)
            return families

        # Compare the object color families and the condition color families
        obj_color_families = get_color_families(closest_name)

        # Check if the condition is a valid CSS4 color
        if condition not in mcolors.CSS4_COLORS:
            raise ValueError(f"Condition color '{condition}' is not a valid CSS4 color.")
        
        # Use the condition as is, no need to find the closest color
        condition_color_families = get_color_families(condition)


        # print("obj_color_families: ", obj_color_families)
        # print("condition_color_families: ", condition_color_families)

        # Return 1 if any family from the object color matches with any family from the condition color
        if obj_color == "brown":
            return True
        for obj_family in obj_color_families:
            for condition_family in condition_color_families:
                if obj_family == condition_family:
                    return True
        return False
        """
"""
@register_score_func
class ScoreColor(ScoreFuncBase):
    NAME = "color"
    NEED_ANCHOR = False  # anchor는 더 이상 필요하지 않음

    @staticmethod
    def get_scores(
        candidate_instances: list['ObjInstance'],
        condition: str
    ) -> float:

        def circular_distance(hue1, hue2):
            hue_diff = abs(hue1 - hue2) 
            return min(hue_diff, 1 - hue_diff) * 2

        def circular_mean(hues):
            angles = 2 * np.pi * hues  # Convert hue values to angles
            mean_angle = np.arctan2(np.mean(np.sin(angles)), np.mean(np.cos(angles)))  # Calculate mean angle
            return (mean_angle / (2 * np.pi)) % 1  # Normalize back to [0, 1] range


        # Define a custom distance function for circular hue clustering
        def custom_distance(x, y):
            return circular_distance(x[0], y[0])
    
        # Perform K-means clustering on hue values using the custom distance function
        class CustomKMeans(KMeans):
            def fit(self, X):
                self.n_samples_ = X.shape[0]
                self.labels_ = np.zeros(self.n_samples_, dtype=int)
                self.cluster_centers_ = np.random.rand(self.n_clusters, 1)  # Initialize centers randomly (only Hue)

                for _ in range(40):  # Arbitrary number of iterations for convergence
                    # Compute distances between each point and cluster centers
                    distances = np.array([[custom_distance(X[i], self.cluster_centers_[j]) 
                                        for j in range(self.n_clusters)] 
                                        for i in range(self.n_samples_)])
                    
                    # Assign each point to the nearest cluster
                    self.labels_ = np.argmin(distances, axis=1)

                    # Update cluster centers using circular mean
                    for j in range(self.n_clusters):
                        points_in_cluster = X[self.labels_ == j]
                        if points_in_cluster.shape[0] > 0:
                            self.cluster_centers_[j, 0] = circular_mean(points_in_cluster[:, 0])

                return self

        # 색상 이름을 RGB로 변환
        try:
            condition_rgb = webcolors.name_to_rgb(condition)
            condition_color = (condition_rgb.red, condition_rgb.green, condition_rgb.blue)
        except ValueError:
            return 1.0

        obj = candidate_instances[0]
        obj_vertex_info = []

        if obj.label == "room ceneter":
            return 1.0

        # 객체의 색상 추출
        if hasattr(obj, 'vertices') and obj.vertices.shape[1] >= 6:
            obj_vertex_info = obj.vertices[:, 3:6]
            avg_color = np.mean(obj_vertex_info, axis=0)
            obj_color = tuple(map(int, avg_color * 255))
        else:
            raise ValueError(f"Instance {obj.inst_id} does not have RGB values in vertices.")
        
        
        hue_values = np.array([colorsys.rgb_to_hsv(rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0)[0] for rgb in obj_vertex_info])
        hue_values = hue_values.reshape(-1, 1)  # Reshape to (N,1) for KMeans

        # 색상 유사도 계산 (HSV 변환 후)
        obj_hsv = colorsys.rgb_to_hsv(*(np.array(obj_color) / 255.0))
        condition_hsv = colorsys.rgb_to_hsv(*(np.array(condition_color) / 255.0))

        if (condition_hsv[1] < 0.1):
            if condition == "white" and obj_hsv[2] >= 0.5 and obj_hsv[1] <= 0.3:
                return 1.0
            elif condition == "black" and obj_hsv[2] <= 0.5:
                return 1.0
            elif obj_hsv[1] <= 0.3:
                return 1.0
            
        #if condition != "red" and condition != "green" and condition != "blue":
        #    hue_diff = circular_distance(condition_hsv[0], obj_hsv[0])
        #    return float(max(0, 1 - hue_diff))
        
        if obj.clustering_hue is not None:
            hue_diff = circular_distance(condition_hsv[0], obj.clustering_hue)
            return float(max(0, 1 - hue_diff))
    
        # Perform K-means clustering on hue values
        k = 10
        kmeans = CustomKMeans(n_clusters=k, random_state=0, n_init=10)
        kmeans.fit(hue_values)

        # Get cluster centers (only hue values)
        cluster_centers = kmeans.cluster_centers_

        # Calculate cluster sizes
        labels = kmeans.labels_
        unique, counts = np.unique(labels, return_counts=True)
        cluster_sizes = dict(zip(unique, counts))

        # Compute weighted average hue
        total_vertices = len(hue_values)
        min_cluster_size = total_vertices / k / 2

        filtered_centers = [(cluster_idx, cluster_centers[cluster_idx]) 
                            for cluster_idx, size in cluster_sizes.items() if size >= min_cluster_size]

        weighted_hues = np.array([filtered_centers[i][1][0] * cluster_sizes[filtered_centers[i][0]] 
                                for i in range(len(filtered_centers))])
        total_weight = np.sum([cluster_sizes[filtered_centers[i][0]] for i in range(len(filtered_centers))])
        avg_hue = np.sum(weighted_hues) / total_weight

        obj.clustering_hue = avg_hue
        hue_diff = circular_distance(condition_hsv[0], obj.clustering_hue)

        print("condition, kmean, non: ", condition, hue_diff, circular_distance(condition_hsv[0], obj_hsv[0]))
        return float(max(0, 1 - hue_diff))
    
"""
@register_score_func
class ScoreColor(ScoreFuncBase):
    NAME = "color"
    NEED_ANCHOR = False 

    @staticmethod
    def get_scores(
        candidate_instances: list['ObjInstance'],
        condition: str
    ) -> float:

        def circular_distance(hue1, hue2):
            hue_diff = abs(hue1 - hue2) 
            return min(hue_diff, 1 - hue_diff) * 2

        # 색상 이름을 RGB로 변환
        try:
            condition_rgb = webcolors.name_to_rgb(condition)
            condition_color = (condition_rgb.red, condition_rgb.green, condition_rgb.blue)
        except Exception:
            return 1.0

        obj = candidate_instances[0]
        obj_vertex_info = []

        if int(obj.inst_id) < 0:
            return 1.0
        
        if hasattr(obj, 'vertices') and obj.vertices.shape[1] >= 6:
            obj_vertex_info = obj.vertices[:, 3:6]
            avg_color = np.mean(obj_vertex_info, axis=0)
            obj_color = tuple(map(int, avg_color * 255))
        else:
            raise ValueError(f"Instance {obj.inst_id} does not have RGB values in vertices.")

        obj_hsv = colorsys.rgb_to_hsv(*(np.array(obj_color) / 255.0))
        condition_hsv = colorsys.rgb_to_hsv(*(np.array(condition_color) / 255.0))

        if condition_hsv[1] < 0.1:
            if condition == "white" and obj_hsv[2] >= 0.5 and obj_hsv[1] <= 0.3:
                return 1.0
            if condition == "black" and obj_hsv[2] <= 0.5:
                return 1.0
            if obj_hsv[1] <= 0.3:
                return 1.0

        #hue_diff = circular_distance(condition_hsv[0], obj_hsv[0])
        #return float(max(0, 1 - hue_diff))

        if obj.clustering_hue is not None:
            hue_diff = circular_distance(condition_hsv[0], obj.clustering_hue)
            return float(max(0, 1 - hue_diff))
        
        
        hue_values = []
        for vertex in obj_vertex_info:
            hsv = colorsys.rgb_to_hsv(vertex[0] / 255.0, vertex[1] / 255.0, vertex[2] / 255.0)
            hue_values.append(hsv[0])
        hue_values = np.array(hue_values).reshape(-1, 1)

        """     
        from sklearn.neighbors import NearestNeighbors
        import matplotlib.pyplot as plt
        neighbors = NearestNeighbors(n_neighbors=max(4, int(np.log(len(hue_values)))))
        neighbors.fit(hue_values)
        distances, indices = neighbors.kneighbors(hue_values)

        # k-dist 플롯: 각 포인트의 k번째 이웃과의 거리 계산
        k_distances = np.sort(distances[:, -1])
        plt.plot(k_distances)
        plt.title("k-dist Plot")
        plt.xlabel("Points")
        plt.ylabel("Distance to kth nearest neighbor")
        plt.ylim(0, 0.01)
        plt.yticks(np.arange(0, 0.01, 0.001))
        plt.savefig("k_dist_plot.png")
        """        

        # Perform DBSCAN clustering on hue values
        dbscan = DBSCAN(eps=0.003, min_samples=max(4, int(np.log(len(hue_values)))), metric='euclidean')
        labels = dbscan.fit_predict(hue_values)

        # 유효한 클러스터가 없으면 최대 유사도 반환
        if len(set(labels)) == 1 and -1 in labels:
            obj.clustering_hue = 1.0
            return 1.0  

        valid_vertices = obj_vertex_info[labels != -1]
        valid_rgb = np.mean(valid_vertices, axis=0) * 255
        obj_color = tuple(map(int, valid_rgb))
        avg_hue = colorsys.rgb_to_hsv(*(np.array(obj_color) / 255.0))

        # obj에 avg_hue 값 저장
        obj.clustering_hue = avg_hue[0]

        num_excluded = np.sum(labels == -1)
        total_data = len(hue_values)

        # hue 차이 계산
        hue_diff = circular_distance(condition_hsv[0], obj.clustering_hue)

        return float(max(0, 1 - hue_diff))