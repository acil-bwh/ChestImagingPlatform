import numpy as np
import SimpleITK as sitk
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize
from scipy.ndimage import maximum_filter

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from scipy.ndimage import (
    map_coordinates,
    label,
)
from scipy.spatial import cKDTree

from collections import deque

import argparse
import csv
import os
import sys

#Labels Extraparenchymal airways

EXT_TRACHEA = 1
EXT_RIGHT_UPPER = 2
EXT_RIGHT_LOWER = 3
EXT_LEFT_UPPER = 4
EXT_LEFT_LOWER = 5


EXTERNAL_REGION_LABELS = {
    "RUL": [
        EXT_TRACHEA,
        EXT_RIGHT_UPPER,
    ],

    "RML": [
        EXT_TRACHEA,
        EXT_RIGHT_LOWER,
    ],

    "RLL": [
        EXT_TRACHEA,
        EXT_RIGHT_LOWER,
    ],

    "LUL": [
        EXT_TRACHEA,
        EXT_LEFT_UPPER,
    ],

    "LLL": [
        EXT_TRACHEA,
        EXT_LEFT_LOWER,
    ],

    "RIGHT": [
        EXT_TRACHEA,
        EXT_RIGHT_UPPER,
        EXT_RIGHT_LOWER,
    ],

    "LEFT": [
        EXT_TRACHEA,
        EXT_LEFT_UPPER,
        EXT_LEFT_LOWER,
    ],

    "WHOLE": [
        EXT_TRACHEA,
        EXT_RIGHT_UPPER,
        EXT_RIGHT_LOWER,
        EXT_LEFT_UPPER,
        EXT_LEFT_LOWER,
    ],
}



##Helper functions

def check_same_geometry(img1, img2, name1="image1", name2="image2"):
    """
    Verify that two SimpleITK images share the same voxel geometry.
    """

    if img1.GetSize() != img2.GetSize():
        raise ValueError(
            f"{name1} and {name2} have different sizes: "
            f"{img1.GetSize()} vs {img2.GetSize()}"
        )

    if not np.allclose(
        img1.GetSpacing(),
        img2.GetSpacing(),
        atol=1e-5,
    ):
        raise ValueError(
            f"{name1} and {name2} have different spacing."
        )

    if not np.allclose(
        img1.GetOrigin(),
        img2.GetOrigin(),
        atol=1e-4,
    ):
        raise ValueError(
            f"{name1} and {name2} have different origins."
        )

    if not np.allclose(
        img1.GetDirection(),
        img2.GetDirection(),
        atol=1e-5,
    ):
        raise ValueError(
            f"{name1} and {name2} have different directions."
        )


def make_binary_region(label_img, labels):
    """
    Create binary region from one or more integer labels.

    Example
    -------
    make_binary_region(lobes_img, [1, 2, 3])
    """

    arr = sitk.GetArrayFromImage(label_img)

    mask = np.isin(
        arr,
        labels
    ).astype(np.uint8)

    out = sitk.GetImageFromArray(mask)
    out.CopyInformation(label_img)

    return out


def parse_region_definition(text):
    """
    Parse region definition:

        NAME:1
        NAME:1,2,3

    Examples
    --------
    RUL:1
    RIGHT:1,2,3
    WHOLE:1,2,3,4,5
    """

    try:

        name, label_text = text.split(
            ":",
            maxsplit=1
        )

        labels = [
            int(x)
            for x in label_text.split(",")
        ]

    except Exception as exc:

        raise argparse.ArgumentTypeError(
            "Region must have format NAME:label[,label,...], "
            "for example RUL:1 or RIGHT:1,2,3"
        ) from exc

    name = name.strip()

    if not name:
        raise argparse.ArgumentTypeError(
            "Region name cannot be empty."
        )

    return name, labels


def compute_voronoi_region_map(
    region_label_img,
    support_mask_img=None,
    background_label=0,
    max_assignment_distance_mm=20,
):
    """
    Voronoi-expand a multi-label anatomical segmentation.

    Voxels are assigned to the closest non-background region.

    Optionally, assignments farther than
    max_assignment_distance_mm from the true anatomy are rejected.
    """

    labels = sitk.GetArrayFromImage(
        region_label_img
    )

    spacing_xyz = np.asarray(
        region_label_img.GetSpacing(),
        dtype=np.float64
    )
    spacing_zyx = spacing_xyz[::-1]

    region = (
        labels != background_label
    )

    if not np.any(region):
        raise ValueError(
            "region_label_img contains no labeled regions."
        )

    # -------------------------------------------------------
    # Binary foreground image
    #
    # Danielsson computes distance to non-zero voxels.
    # -------------------------------------------------------

    binary_region_img = sitk.Cast(
        region_label_img != background_label,
        sitk.sitkUInt8,
    )

    # -------------------------------------------------------
    # Danielsson distance transform
    # -------------------------------------------------------

    print("Computing Danielsson distance map...")

    danielsson = (
        sitk.DanielssonDistanceMapImageFilter()
    )
    danielsson.SetInputIsBinary(True)
    danielsson.SetSquaredDistance(False)
    danielsson.SetUseImageSpacing(True)
    distance_img = danielsson.Execute(
        binary_region_img
    )

    # -------------------------------------------------------
    # Vector map
    #
    # Each voxel contains the integer index displacement
    # to the closest foreground voxel.
    # -------------------------------------------------------

    vector_img = danielsson.GetVectorDistanceMap()
    print("Danielsson distance map complete.")

    # -------------------------------------------------------
    # Convert to NumPy
    # -------------------------------------------------------
    distance_mm = (
        sitk.GetArrayViewFromImage(
            distance_img
        )
    )
    vectors = (
        sitk.GetArrayViewFromImage(
            vector_img
        )
    )

    # SimpleITK vector components are x, y, z.
    #
    # NumPy image indexing is z, y, x.

    dx = vectors[..., 0]
    dy = vectors[..., 1]
    dz = vectors[..., 2]

    # -------------------------------------------------------
    # Construct voxel coordinates
    # -------------------------------------------------------
    nz, ny, nx = labels.shape
    z, y, x = np.indices(
        labels.shape,
        dtype=np.int32,
        sparse=True,
    )

    nearest_x = x + dx
    nearest_y = y + dy
    nearest_z = z + dz

    # -------------------------------------------------------
    # Safety clipping
    # -------------------------------------------------------
    nearest_x = np.clip(
        nearest_x,
        0,
        nx - 1,
    )

    nearest_y = np.clip(
        nearest_y,
        0,
        ny - 1,
    )

    nearest_z = np.clip(
        nearest_z,
        0,
        nz - 1,
    )

    # -------------------------------------------------------
    # Propagate nearest anatomical label
    # -------------------------------------------------------

    expanded = labels[
        nearest_z,
        nearest_y,
        nearest_x,
    ].copy()


    # -------------------------------------------------------
    # Remove assignments farther than allowed distance
    # -------------------------------------------------------

    if max_assignment_distance_mm is not None:

        expanded[
            distance_mm >
            max_assignment_distance_mm
        ] = background_label



    # ----------------------------------------
    # Optional external support constraint
    # ----------------------------------------

    if support_mask_img is not None:

        support = (
            sitk.GetArrayFromImage(
                support_mask_img
            ) > 0
        )

        expanded[~support] = background_label

    # True anatomy always retains its label
    expanded[region] = labels[region]

    out = sitk.GetImageFromArray(
        expanded.astype(labels.dtype)
    )

    out.CopyInformation(
        region_label_img
    )

    return out


#Compute external airway labels using plain K-means
def compute_external_airway_labels_v0(
    skeleton_img,
    region_label_img,
    rul_label=4,
    rml_label=5,
    rll_label=6,
    lul_label=7,
    lll_label=8,
    random_state=0,
):
    """
    Classify extraparenchymal airway skeleton voxels into:

        1 = trachea
        2 = right upper pathway
        3 = right lower pathway
        4 = left upper pathway
        5 = left lower pathway

    Classification is hierarchical:

        Level 1:
            trachea / right / left

        Level 2:
            right -> upper / lower
            left  -> upper / lower

    K-means operates on physical coordinates. Anatomical lobe
    centroids are then used to assign semantic identities to the
    resulting clusters.

    Parameters
    ----------
    skeleton_img : sitk.Image
        Binary airway centerline image.

    region_label_img : sitk.Image
        Lobar label map.

    Returns
    -------
    sitk.Image
        UInt8 image containing external-airway labels only.

    dict
        Diagnostic information describing cluster sizes and
        centroids.
    """

    # ---------------------------------------------------------
    # Input arrays
    # ---------------------------------------------------------

    skeleton = (
        sitk.GetArrayViewFromImage(skeleton_img) > 0
    )

    lobes = sitk.GetArrayViewFromImage(
        region_label_img
    )

    whole_lung = (
        (lobes == rul_label) |
        (lobes == rml_label) |
        (lobes == rll_label) |
        (lobes == lul_label) |
        (lobes == lll_label)
    )

    external_skeleton = (
        skeleton & ~whole_lung
    )

    external_indices_zyx = np.argwhere(
        external_skeleton
    )

    if len(external_indices_zyx) < 10:
        raise ValueError(
            "Too few extraparenchymal skeleton voxels."
        )

    # ---------------------------------------------------------
    # Convert NumPy indices to physical coordinates.
    #
    # SimpleITK uses x,y,z.
    # NumPy uses z,y,x.
    # ---------------------------------------------------------

    external_points_xyz = np.asarray([
        skeleton_img.TransformIndexToPhysicalPoint(
            (
                int(x),
                int(y),
                int(z),
            )
        )
        for z, y, x in external_indices_zyx
    ])

    # ---------------------------------------------------------
    # Compute physical centroids of each lobe
    # ---------------------------------------------------------

    def physical_centroid(label):

        coords_zyx = np.argwhere(
            lobes == label
        )

        if len(coords_zyx) == 0:
            raise ValueError(
                f"Lobe label {label} is absent."
            )

        centroid_zyx = np.mean(
            coords_zyx,
            axis=0,
        )

        # Convert continuous NumPy z,y,x coordinate to
        # SimpleITK continuous x,y,z index.

        centroid_xyz_index = (
            float(centroid_zyx[2]),
            float(centroid_zyx[1]),
            float(centroid_zyx[0]),
        )

        return np.asarray(
            region_label_img.TransformContinuousIndexToPhysicalPoint(
                centroid_xyz_index
            ),
            dtype=np.float64,
        )

    C_RUL = physical_centroid(rul_label)
    C_RML = physical_centroid(rml_label)
    C_RLL = physical_centroid(rll_label)

    C_LUL = physical_centroid(lul_label)
    C_LLL = physical_centroid(lll_label)

    # Side-level targets

    C_RIGHT = np.mean(
        [C_RUL, C_RML, C_RLL],
        axis=0,
    )

    C_LEFT = np.mean(
        [C_LUL, C_LLL],
        axis=0,
    )

    # Right lower target

    C_RIGHT_LOWER = np.mean(
        [C_RML, C_RLL],
        axis=0,
    )

    # ---------------------------------------------------------
    # LEVEL 1
    #
    # External skeleton ->
    # trachea / right / left
    # ---------------------------------------------------------

    kmeans_level1 = KMeans(
        n_clusters=3,
        random_state=random_state,
        n_init=20,
    )

    level1 = kmeans_level1.fit_predict(
        external_points_xyz
    )

    centers1 = kmeans_level1.cluster_centers_

    # ---------------------------------------------------------
    # Identify RIGHT and LEFT clusters from lobe centroids.
    # ---------------------------------------------------------

    distance_to_right = np.linalg.norm(
        centers1 - C_RIGHT,
        axis=1,
    )

    distance_to_left = np.linalg.norm(
        centers1 - C_LEFT,
        axis=1,
    )

    # Find best distinct assignment of one cluster to right
    # and another to left.

    best_cost = np.inf
    right_cluster = None
    left_cluster = None

    for r in range(3):
        for l in range(3):

            if r == l:
                continue

            cost = (
                distance_to_right[r]
                + distance_to_left[l]
            )

            if cost < best_cost:

                best_cost = cost
                right_cluster = r
                left_cluster = l

    trachea_cluster = next(
        c for c in range(3)
        if c not in (
            right_cluster,
            left_cluster,
        )
    )

    # ---------------------------------------------------------
    # Output labels
    # ---------------------------------------------------------

    external_labels = np.zeros(
        len(external_points_xyz),
        dtype=np.uint8,
    )

    external_labels[
        level1 == trachea_cluster
    ] = EXT_TRACHEA

    # ---------------------------------------------------------
    # LEVEL 2 — RIGHT
    # ---------------------------------------------------------

    right_mask = (
        level1 == right_cluster
    )

    right_points = external_points_xyz[
        right_mask
    ]

    if len(right_points) >= 2:

        kmeans_right = KMeans(
            n_clusters=2,
            random_state=random_state,
            n_init=20,
        )

        right_sub = (
            kmeans_right.fit_predict(
                right_points
            )
        )

        right_centers = (
            kmeans_right.cluster_centers_
        )

        # Assign upper/lower semantic identity

        d_upper = np.linalg.norm(
            right_centers - C_RUL,
            axis=1,
        )

        d_lower = np.linalg.norm(
            right_centers - C_RIGHT_LOWER,
            axis=1,
        )

        # Determine the best distinct assignment

        cost_01 = (
            d_upper[0] +
            d_lower[1]
        )

        cost_10 = (
            d_upper[1] +
            d_lower[0]
        )

        if cost_01 <= cost_10:
            upper_cluster = 0
            lower_cluster = 1
        else:
            upper_cluster = 1
            lower_cluster = 0

        right_indices = np.where(
            right_mask
        )[0]

        external_labels[
            right_indices[
                right_sub == upper_cluster
            ]
        ] = EXT_RIGHT_UPPER

        external_labels[
            right_indices[
                right_sub == lower_cluster
            ]
        ] = EXT_RIGHT_LOWER

    # ---------------------------------------------------------
    # LEVEL 2 — LEFT
    # ---------------------------------------------------------

    left_mask = (
        level1 == left_cluster
    )

    left_points = external_points_xyz[
        left_mask
    ]

    if len(left_points) >= 2:

        kmeans_left = KMeans(
            n_clusters=2,
            random_state=random_state,
            n_init=20,
        )

        left_sub = (
            kmeans_left.fit_predict(
                left_points
            )
        )

        left_centers = (
            kmeans_left.cluster_centers_
        )

        d_upper = np.linalg.norm(
            left_centers - C_LUL,
            axis=1,
        )

        d_lower = np.linalg.norm(
            left_centers - C_LLL,
            axis=1,
        )

        cost_01 = (
            d_upper[0] +
            d_lower[1]
        )

        cost_10 = (
            d_upper[1] +
            d_lower[0]
        )

        if cost_01 <= cost_10:
            upper_cluster = 0
            lower_cluster = 1
        else:
            upper_cluster = 1
            lower_cluster = 0

        left_indices = np.where(
            left_mask
        )[0]

        external_labels[
            left_indices[
                left_sub == upper_cluster
            ]
        ] = EXT_LEFT_UPPER

        external_labels[
            left_indices[
                left_sub == lower_cluster
            ]
        ] = EXT_LEFT_LOWER

    # ---------------------------------------------------------
    # Build output image
    # ---------------------------------------------------------

    label_map = np.zeros(
        skeleton.shape,
        dtype=np.uint8,
    )

    z = external_indices_zyx[:, 0]
    y = external_indices_zyx[:, 1]
    x = external_indices_zyx[:, 2]

    label_map[z, y, x] = external_labels

    out = sitk.GetImageFromArray(
        label_map
    )

    out.CopyInformation(
        skeleton_img
    )

    # ---------------------------------------------------------
    # Diagnostics
    # ---------------------------------------------------------

    diagnostics = {
        "n_external_points":
            len(external_points_xyz),

        "n_trachea":
            int(np.sum(
                external_labels == EXT_TRACHEA
            )),

        "n_right_upper":
            int(np.sum(
                external_labels == EXT_RIGHT_UPPER
            )),

        "n_right_lower":
            int(np.sum(
                external_labels == EXT_RIGHT_LOWER
            )),

        "n_left_upper":
            int(np.sum(
                external_labels == EXT_LEFT_UPPER
            )),

        "n_left_lower":
            int(np.sum(
                external_labels == EXT_LEFT_LOWER
            )),

        "lobe_centroids": {
            "RUL": C_RUL,
            "RML": C_RML,
            "RLL": C_RLL,
            "LUL": C_LUL,
            "LLL": C_LLL,
        },

        "level1_centers":
            centers1,
    }

    return out, diagnostics



#Compute external airway labels using a Kmeans that
#Users coordinates and angle from centroid.

def compute_external_airway_labels(
    skeleton_img,
    region_label_img,
    rul_label=4,
    rml_label=5,
    rll_label=6,
    lul_label=7,
    lll_label=8,
    random_state=0,
    n_init=20,
):
    """
    Hierarchically classify extraparenchymal airway skeleton voxels.

    Output classes
    --------------
    0 : unlabeled / not external airway
    1 : trachea
    2 : right upper pathway
    3 : right lower pathway
    4 : left upper pathway
    5 : left lower pathway

    Strategy
    --------
    1. Identify airway skeleton voxels outside the whole-lung mask.

    2. Level-1 K-means:
           [x, y, ux, uy]

       where x,y are physical coordinates and ux,uy are the
       transverse components of the unit vector from the central
       thoracic reference point to the airway point.

       K=3:
           trachea / right / left

       The right and left clusters are identified by proximity
       to the corresponding lung centroids. The remaining cluster
       is assigned to the trachea.

    3. Level-2 K-means separately within right and left clusters:
           [x, y, z, ux, uy, uz]

       K=2 on each side:
           upper / lower

       Cluster identities are assigned using the corresponding
       lobar centroids.

    Notes
    -----
    All coordinates are physical coordinates (mm).

    Features are standardized before K-means so that spatial and
    directional components contribute comparably.
    """

    # ========================================================
    # Input arrays
    # ========================================================

    skeleton = (
        sitk.GetArrayViewFromImage(skeleton_img) > 0
    )

    lobes = sitk.GetArrayViewFromImage(
        region_label_img
    )

    if skeleton.shape != lobes.shape:
        raise ValueError(
            "skeleton_img and region_label_img "
            "must have the same image dimensions."
        )

    lobe_labels = [
        rul_label,
        rml_label,
        rll_label,
        lul_label,
        lll_label,
    ]

    whole_lung = np.isin(
        lobes,
        lobe_labels,
    )

    # --------------------------------------------------------
    # External airway skeleton
    # --------------------------------------------------------

    external_skeleton = (
        skeleton & ~whole_lung
    )

    indices_zyx = np.argwhere(
        external_skeleton
    )

    n_external = len(indices_zyx)

    if n_external < 10:
        raise ValueError(
            "Too few extraparenchymal airway skeleton "
            f"voxels ({n_external})."
        )

    # ========================================================
    # Helper: physical centroid of an anatomical label
    # ========================================================

    def get_label_centroid(label):

        coords_zyx = np.argwhere(
            lobes == label
        )

        if len(coords_zyx) == 0:
            raise ValueError(
                f"Region label {label} is absent."
            )

        centroid_zyx = np.mean(
            coords_zyx,
            axis=0,
        )

        continuous_index_xyz = (
            float(centroid_zyx[2]),
            float(centroid_zyx[1]),
            float(centroid_zyx[0]),
        )

        return np.asarray(
            region_label_img.
            TransformContinuousIndexToPhysicalPoint(
                continuous_index_xyz
            ),
            dtype=np.float64,
        )

    # ========================================================
    # Anatomical centroids
    # ========================================================

    C_RUL = get_label_centroid(rul_label)
    C_RML = get_label_centroid(rml_label)
    C_RLL = get_label_centroid(rll_label)

    C_LUL = get_label_centroid(lul_label)
    C_LLL = get_label_centroid(lll_label)

    # --------------------------------------------------------
    # Side centroids
    # --------------------------------------------------------

    C_RIGHT = np.mean(
        np.vstack([
            C_RUL,
            C_RML,
            C_RLL,
        ]),
        axis=0,
    )

    C_LEFT = np.mean(
        np.vstack([
            C_LUL,
            C_LLL,
        ]),
        axis=0,
    )

    # --------------------------------------------------------
    # Central thoracic reference point
    # --------------------------------------------------------

    origin = (
        C_RIGHT + C_LEFT
    ) / 2.0

    # --------------------------------------------------------
    # Right lower target
    # --------------------------------------------------------

    C_RIGHT_LOWER = np.mean(
        np.vstack([
            C_RML,
            C_RLL,
        ]),
        axis=0,
    )

    # ========================================================
    # Convert external skeleton indices to physical coordinates
    # ========================================================

    points_xyz = np.asarray(
        [
            skeleton_img.TransformIndexToPhysicalPoint(
                (
                    int(x),
                    int(y),
                    int(z),
                )
            )
            for z, y, x in indices_zyx
        ],
        dtype=np.float64,
    )

    # ========================================================
    # Direction vectors from central thoracic reference
    # ========================================================

    vectors_xyz = (
        points_xyz - origin
    )

    norms = np.linalg.norm(
        vectors_xyz,
        axis=1,
        keepdims=True,
    )

    # Avoid division by zero
    norms = np.maximum(
        norms,
        1e-6,
    )

    unit_xyz = (
        vectors_xyz / norms
    )

    # ========================================================
    # LEVEL 1
    #
    # Trachea / Right / Left
    #
    # Deliberately exclude Z to reduce the tendency of
    # K-means to split the long trachea superior-inferiorly.
    # ========================================================

    level1_features = np.column_stack([
        points_xyz[:, 0],  # X
        points_xyz[:, 1],  # Y
        unit_xyz[:, 0],    # Ux
        unit_xyz[:, 1],    # Uy
    ])

    level1_scaler = StandardScaler()

    level1_features_scaled = (
        level1_scaler.fit_transform(
            level1_features
        )
    )

    kmeans_level1 = KMeans(
        n_clusters=3,
        n_init=n_init,
        random_state=random_state,
    )

    level1_labels = (
        kmeans_level1.fit_predict(
            level1_features_scaled
        )
    )

    # ========================================================
    # Compute Level-1 cluster centroids in ORIGINAL
    # physical XYZ coordinates.
    #
    # Do NOT use the standardized feature-space centroids for
    # anatomical identification.
    # ========================================================

    level1_centers_xyz = np.zeros(
        (3, 3),
        dtype=np.float64,
    )

    for cluster_id in range(3):

        mask = (
            level1_labels == cluster_id
        )

        level1_centers_xyz[
            cluster_id
        ] = np.mean(
            points_xyz[mask],
            axis=0,
        )

    # ========================================================
    # Identify RIGHT and LEFT Level-1 clusters
    # ========================================================

    distance_to_right = np.linalg.norm(
        level1_centers_xyz - C_RIGHT,
        axis=1,
    )

    distance_to_left = np.linalg.norm(
        level1_centers_xyz - C_LEFT,
        axis=1,
    )

    # Find the best distinct right/left assignment.
    best_cost = np.inf

    right_cluster = None
    left_cluster = None

    for r in range(3):

        for l in range(3):

            if r == l:
                continue

            cost = (
                distance_to_right[r]
                +
                distance_to_left[l]
            )

            if cost < best_cost:

                best_cost = cost

                right_cluster = r
                left_cluster = l

    # The remaining cluster is trachea.
    trachea_cluster = next(
        c
        for c in range(3)
        if c not in (
            right_cluster,
            left_cluster,
        )
    )

    # ========================================================
    # Final point labels
    # ========================================================

    external_labels = np.zeros(
        n_external,
        dtype=np.uint8,
    )

    external_labels[
        level1_labels == trachea_cluster
    ] = EXT_TRACHEA

    # ========================================================
    # Helper for Level-2 subdivision
    # ========================================================

    def subdivide_side(
        side_mask,
        upper_target,
        lower_target,
        upper_output_label,
        lower_output_label,
    ):
        """
        Split one side into upper and lower pathways using
        six-dimensional XYZ + directional features.
        """

        point_indices = np.where(
            side_mask
        )[0]

        side_points = points_xyz[
            point_indices
        ]

        side_units = unit_xyz[
            point_indices
        ]

        if len(side_points) < 2:

            return {
                "upper_center": None,
                "lower_center": None,
                "n_upper": 0,
                "n_lower": 0,
            }

        # ----------------------------------------------------
        # Six-dimensional feature space
        # ----------------------------------------------------

        features = np.column_stack([
            side_points[:, 0],  # X
            side_points[:, 1],  # Y
            side_points[:, 2],  # Z

            side_units[:, 0],   # Ux
            side_units[:, 1],   # Uy
            side_units[:, 2],   # Uz
        ])

        scaler = StandardScaler()

        features_scaled = (
            scaler.fit_transform(
                features
            )
        )

        kmeans = KMeans(
            n_clusters=2,
            n_init=n_init,
            random_state=random_state,
        )

        sublabels = (
            kmeans.fit_predict(
                features_scaled
            )
        )

        # ----------------------------------------------------
        # Physical XYZ centroid of each subcluster
        # ----------------------------------------------------

        centers_xyz = np.zeros(
            (2, 3),
            dtype=np.float64,
        )

        for cluster_id in range(2):

            cluster_mask = (
                sublabels == cluster_id
            )

            centers_xyz[
                cluster_id
            ] = np.mean(
                side_points[
                    cluster_mask
                ],
                axis=0,
            )

        # ----------------------------------------------------
        # Determine which cluster is upper vs lower.
        # ----------------------------------------------------

        d_upper = np.linalg.norm(
            centers_xyz - upper_target,
            axis=1,
        )

        d_lower = np.linalg.norm(
            centers_xyz - lower_target,
            axis=1,
        )

        # Two possible one-to-one assignments.
        cost_01 = (
            d_upper[0]
            +
            d_lower[1]
        )

        cost_10 = (
            d_upper[1]
            +
            d_lower[0]
        )

        if cost_01 <= cost_10:

            upper_cluster = 0
            lower_cluster = 1

        else:

            upper_cluster = 1
            lower_cluster = 0

        # ----------------------------------------------------
        # Write semantic labels back into global array
        # ----------------------------------------------------

        upper_indices = point_indices[
            sublabels == upper_cluster
        ]

        lower_indices = point_indices[
            sublabels == lower_cluster
        ]

        external_labels[
            upper_indices
        ] = upper_output_label

        external_labels[
            lower_indices
        ] = lower_output_label

        return {
            "upper_center":
                centers_xyz[upper_cluster],

            "lower_center":
                centers_xyz[lower_cluster],

            "n_upper":
                int(len(upper_indices)),

            "n_lower":
                int(len(lower_indices)),
        }

    # ========================================================
    # LEVEL 2 — RIGHT
    # ========================================================

    right_mask = (
        level1_labels == right_cluster
    )

    right_info = subdivide_side(
        side_mask=right_mask,

        upper_target=C_RUL,

        lower_target=C_RIGHT_LOWER,

        upper_output_label=
            EXT_RIGHT_UPPER,

        lower_output_label=
            EXT_RIGHT_LOWER,
    )

    # ========================================================
    # LEVEL 2 — LEFT
    # ========================================================

    left_mask = (
        level1_labels == left_cluster
    )

    left_info = subdivide_side(
        side_mask=left_mask,

        upper_target=C_LUL,

        lower_target=C_LLL,

        upper_output_label=
            EXT_LEFT_UPPER,

        lower_output_label=
            EXT_LEFT_LOWER,
    )

    # ========================================================
    # Construct label image
    # ========================================================

    label_map = np.zeros(
        skeleton.shape,
        dtype=np.uint8,
    )

    z = indices_zyx[:, 0]
    y = indices_zyx[:, 1]
    x = indices_zyx[:, 2]

    label_map[
        z, y, x
    ] = external_labels

    output_img = sitk.GetImageFromArray(
        label_map
    )

    output_img.CopyInformation(
        skeleton_img
    )

    # ========================================================
    # Diagnostics
    # ========================================================

    diagnostics = {

        "n_external_points":
            int(n_external),

        "n_trachea":
            int(np.sum(
                external_labels
                == EXT_TRACHEA
            )),

        "n_right_upper":
            int(np.sum(
                external_labels
                == EXT_RIGHT_UPPER
            )),

        "n_right_lower":
            int(np.sum(
                external_labels
                == EXT_RIGHT_LOWER
            )),

        "n_left_upper":
            int(np.sum(
                external_labels
                == EXT_LEFT_UPPER
            )),

        "n_left_lower":
            int(np.sum(
                external_labels
                == EXT_LEFT_LOWER
            )),

        "origin_xyz":
            origin,

        "right_lung_centroid_xyz":
            C_RIGHT,

        "left_lung_centroid_xyz":
            C_LEFT,

        "RUL_centroid_xyz":
            C_RUL,

        "RML_centroid_xyz":
            C_RML,

        "RLL_centroid_xyz":
            C_RLL,

        "LUL_centroid_xyz":
            C_LUL,

        "LLL_centroid_xyz":
            C_LLL,

        "level1_centers_xyz":
            level1_centers_xyz,

        "level1_right_cluster":
            int(right_cluster),

        "level1_left_cluster":
            int(left_cluster),

        "level1_trachea_cluster":
            int(trachea_cluster),

        "right_subdivision":
            right_info,

        "left_subdivision":
            left_info,
    }

    return output_img, diagnostics



def make_valid_airway_region(
    airway_img,
    region_img,
    external_label_img,
    external_labels,
):
    """
    Combine intraparenchymal airway in the target region
    with selected extraparenchymal airway classes.
    """

    airway = (
        sitk.GetArrayViewFromImage(airway_img) > 0
    )

    region = (
        sitk.GetArrayViewFromImage(region_img) > 0
    )

    ext = sitk.GetArrayViewFromImage(
        external_label_img
    )

    external_selection = np.isin(
        ext,
        external_labels,
    )

    valid = airway & (
        region |
        external_selection
    )

    out = sitk.GetImageFromArray(
        valid.astype(np.uint8)
    )

    out.CopyInformation(
        airway_img
    )

    return out

def make_physical_spherical_footprint(
    spacing_xyz,
    radius_mm=0.75,
):
    """
    Create a boolean footprint corresponding approximately
    to a sphere of the requested physical radius.

    Output is in NumPy z,y,x ordering.
    """

    spacing_zyx = np.asarray(
        spacing_xyz[::-1],
        dtype=np.float64,
    )

    rv = np.ceil(
        radius_mm / spacing_zyx
    ).astype(int)

    z = np.arange(
        -rv[0],
        rv[0] + 1,
    )

    y = np.arange(
        -rv[1],
        rv[1] + 1,
    )

    x = np.arange(
        -rv[2],
        rv[2] + 1,
    )

    zz, yy, xx = np.meshgrid(
        z,
        y,
        x,
        indexing="ij",
    )

    distance_mm = np.sqrt(
        (zz * spacing_zyx[0]) ** 2
        +
        (yy * spacing_zyx[1]) ** 2
        +
        (xx * spacing_zyx[2]) ** 2
    )

    footprint = (
        distance_mm <= radius_mm
    )

    return footprint


def compute_oblique_airway_diameters(
    airway_img,
    skeleton_img,
    edt_radius_img=None,
    tangent_radius_mm=3.0,
    plane_spacing_mm=0.20,
    plane_margin_mm=2.0,
    min_half_width_mm=2.5,
    max_half_width_mm=7.0,
    min_tangent_points=5,
    interpolation_order=1,
    threshold=0.5,
):
    """
    Compute airway lumen diameter at every skeleton point from an
    oblique cross-section perpendicular to the local airway direction.

    The local airway direction is estimated using PCA of neighboring
    skeleton points in physical space.

    The binary airway mask is sampled on the perpendicular plane using
    trilinear interpolation. The lumen is defined using a 0.5 threshold.

    Only the connected component containing the center of the plane is
    used to calculate cross-sectional area.

    The reported diameter is the area-equivalent diameter:

        D_eq = 2 * sqrt(A / pi)

    Parameters
    ----------
    airway_img : sitk.Image
        Binary airway lumen segmentation.

    skeleton_img : sitk.Image
        Binary airway skeleton.

    edt_radius_img : sitk.Image or None
        Optional EDT radius image in mm.

        If supplied, the EDT radius at each skeleton point is used to
        adapt the cross-sectional plane size.

        If None, max_half_width_mm is used for every point.

    tangent_radius_mm : float
        Physical radius around each skeleton point used to estimate
        the local airway tangent by PCA.

    plane_spacing_mm : float
        Sampling resolution of the oblique plane.

    plane_margin_mm : float
        Additional margin added to the EDT radius when determining
        plane size.

    min_half_width_mm : float
        Minimum plane half-width.

    max_half_width_mm : float
        Maximum plane half-width.

    min_tangent_points : int
        Minimum number of neighboring skeleton points required for
        PCA tangent estimation.

    interpolation_order : int
        Interpolation order passed to scipy.ndimage.map_coordinates.
        1 = trilinear interpolation.

    threshold : float
        Threshold applied to the interpolated binary mask.

    Returns
    -------
    dict containing

        diameter_img
            Area-equivalent diameter at each skeleton point.

        area_img
            Cross-sectional area at each skeleton point.

        tangent_quality_img
            PCA linearity metric at each skeleton point.

        valid_img
            Skeleton locations where an oblique measurement succeeded.
    """

    # ============================================================
    # Input arrays
    # ============================================================

    airway = (
        sitk.GetArrayFromImage(airway_img) > 0
    )

    skeleton = (
        sitk.GetArrayFromImage(skeleton_img) > 0
    )

    if airway.shape != skeleton.shape:
        raise ValueError(
            "airway_img and skeleton_img must have identical geometry."
        )

    spacing_xyz = np.asarray(
        airway_img.GetSpacing(),
        dtype=np.float64,
    )

    origin_xyz = np.asarray(
        airway_img.GetOrigin(),
        dtype=np.float64,
    )

    direction = np.asarray(
        airway_img.GetDirection(),
        dtype=np.float64,
    ).reshape(3, 3)

    # Optional EDT radius

    if edt_radius_img is not None:

        edt_radius = np.asarray(
            sitk.GetArrayViewFromImage(edt_radius_img),
            dtype=np.float32,
        )

        if edt_radius.shape != airway.shape:
            raise ValueError(
                "edt_radius_img must match airway_img."
            )

    else:

        edt_radius = None

    # Float image for interpolation

    airway_float = airway.astype(
        np.float32
    )

    # ============================================================
    # Skeleton coordinates
    # ============================================================

    skeleton_zyx = np.argwhere(
        skeleton
    )

    n_points = len(skeleton_zyx)

    if n_points == 0:
        raise ValueError(
            "No skeleton voxels found."
        )

    print(
        f"Computing oblique diameters for "
        f"{n_points} skeleton points"
    )

    # ------------------------------------------------------------
    # Convert z,y,x -> x,y,z index coordinates
    # ------------------------------------------------------------

    skeleton_xyz_index = (
        skeleton_zyx[:, ::-1]
        .astype(np.float64)
    )

    # ------------------------------------------------------------
    # Convert index coordinates to physical coordinates
    #
    # p = origin + direction @ (index * spacing)
    # ------------------------------------------------------------

    scaled = (
        skeleton_xyz_index
        * spacing_xyz
    )

    skeleton_phys = (
        origin_xyz[None, :]
        +
        scaled @ direction.T
    )

    # ============================================================
    # KD-tree for fast physical-neighborhood queries
    # ============================================================

    tree = cKDTree(
        skeleton_phys
    )

    # ============================================================
    # Output arrays
    # ============================================================

    diameter_mm = np.zeros(
        airway.shape,
        dtype=np.float32,
    )

    area_mm2 = np.zeros(
        airway.shape,
        dtype=np.float32,
    )

    tangent_quality = np.zeros(
        airway.shape,
        dtype=np.float32,
    )

    valid = np.zeros(
        airway.shape,
        dtype=bool,
    )

    # ============================================================
    # Precompute inverse direction matrix
    #
    # Needed for:
    # physical point -> continuous image index
    # ============================================================

    inv_direction = np.linalg.inv(
        direction
    )

    # ============================================================
    # Process every skeleton point
    # ============================================================

    for point_number, (
        voxel_zyx,
        center_phys,
    ) in enumerate(
        zip(
            skeleton_zyx,
            skeleton_phys,
        )
    ):

        z, y, x = voxel_zyx

        # --------------------------------------------------------
        # Find nearby skeleton points in physical space
        # --------------------------------------------------------

        neighbor_ids = tree.query_ball_point(
            center_phys,
            r=tangent_radius_mm,
        )

        if len(neighbor_ids) < min_tangent_points:
            continue

        neighbor_points = (
            skeleton_phys[neighbor_ids]
        )

        # --------------------------------------------------------
        # PCA tangent estimation
        # --------------------------------------------------------

        centered = (
            neighbor_points
            -
            neighbor_points.mean(
                axis=0
            )
        )

        covariance = (
            centered.T @ centered
        ) / max(
            len(neighbor_points) - 1,
            1,
        )

        eigenvalues, eigenvectors = (
            np.linalg.eigh(
                covariance
            )
        )

        # eigh returns ascending eigenvalues.

        order = np.argsort(
            eigenvalues
        )[::-1]

        eigenvalues = (
            eigenvalues[order]
        )

        eigenvectors = (
            eigenvectors[:, order]
        )

        tangent = (
            eigenvectors[:, 0]
        )

        tangent /= (
            np.linalg.norm(tangent)
            + 1e-12
        )

        # --------------------------------------------------------
        # PCA linearity / tangent quality
        #
        # 1 -> strongly linear structure
        # 0 -> poorly defined dominant direction
        # --------------------------------------------------------

        if eigenvalues[0] > 0:

            quality = (
                eigenvalues[0]
                -
                eigenvalues[1]
            ) / eigenvalues[0]

        else:

            quality = 0.0

        tangent_quality[
            z, y, x
        ] = quality

        # --------------------------------------------------------
        # Construct orthogonal plane basis
        #
        # Need u perpendicular to tangent.
        # Choose reference axis least aligned with tangent.
        # --------------------------------------------------------

        reference_axes = np.eye(3)

        alignment = np.abs(
            reference_axes @ tangent
        )

        reference = (
            reference_axes[
                np.argmin(alignment)
            ]
        )

        u = np.cross(
            tangent,
            reference,
        )

        u /= (
            np.linalg.norm(u)
            + 1e-12
        )

        v = np.cross(
            tangent,
            u,
        )

        v /= (
            np.linalg.norm(v)
            + 1e-12
        )

        # --------------------------------------------------------
        # Determine plane size
        # --------------------------------------------------------

        if edt_radius is not None:

            radius_estimate = float(
                edt_radius[
                    z, y, x
                ]
            )

            half_width_mm = (
                radius_estimate
                +
                plane_margin_mm
            )

            half_width_mm = np.clip(
                half_width_mm,
                min_half_width_mm,
                max_half_width_mm,
            )

        else:

            half_width_mm = (
                max_half_width_mm
            )

        # --------------------------------------------------------
        # Plane coordinates
        #
        # Make an odd number of samples so the centerline
        # location is exactly the center pixel.
        # --------------------------------------------------------

        n_half = int(
            np.ceil(
                half_width_mm
                /
                plane_spacing_mm
            )
        )

        offsets = (
            np.arange(
                -n_half,
                n_half + 1,
                dtype=np.float64,
            )
            *
            plane_spacing_mm
        )

        aa, bb = np.meshgrid(
            offsets,
            offsets,
            indexing="xy",
        )

        # --------------------------------------------------------
        # Physical coordinates of every plane sample
        #
        # X = center + a*u + b*v
        # --------------------------------------------------------

        plane_phys = (
            center_phys[None, None, :]
            +
            aa[..., None] * u
            +
            bb[..., None] * v
        )

        plane_shape = (
            plane_phys.shape[:2]
        )

        plane_phys_flat = (
            plane_phys.reshape(
                -1,
                3,
            )
        )

        # --------------------------------------------------------
        # Physical coordinates -> continuous xyz image indices
        #
        # index =
        #   inv(direction) @ (physical-origin) / spacing
        # --------------------------------------------------------

        relative = (
            plane_phys_flat
            -
            origin_xyz[None, :]
        )

        continuous_xyz = (
            relative
            @ inv_direction.T
        )

        continuous_xyz /= (
            spacing_xyz[None, :]
        )

        # --------------------------------------------------------
        # scipy wants z,y,x coordinate ordering
        # --------------------------------------------------------

        continuous_zyx = (
            continuous_xyz[:, ::-1]
        )

        coordinates = np.vstack(
            [
                continuous_zyx[:, 0],
                continuous_zyx[:, 1],
                continuous_zyx[:, 2],
            ]
        )

        # --------------------------------------------------------
        # Trilinear interpolation of binary airway mask
        # --------------------------------------------------------

        plane_values = map_coordinates(
            airway_float,
            coordinates,
            order=interpolation_order,
            mode="constant",
            cval=0.0,
            prefilter=False,
        )

        plane_values = (
            plane_values.reshape(
                plane_shape
            )
        )

        # --------------------------------------------------------
        # Threshold interpolated occupancy field
        # --------------------------------------------------------

        plane_mask = (
            plane_values
            >=
            threshold
        )

        # --------------------------------------------------------
        # Connected components
        #
        # Keep only lumen connected to plane center.
        # --------------------------------------------------------

        component_labels, n_components = (
            label(
                plane_mask,
                structure=np.ones(
                    (3, 3),
                    dtype=np.uint8,
                ),
            )
        )

        center_index = n_half

        center_label = (
            component_labels[
                center_index,
                center_index,
            ]
        )

        if center_label == 0:
            continue

        lumen_component = (
            component_labels
            ==
            center_label
        )

        # --------------------------------------------------------
        # Cross-sectional area
        # --------------------------------------------------------

        n_lumen_pixels = np.count_nonzero(
            lumen_component
        )

        area = (
            n_lumen_pixels
            *
            plane_spacing_mm
            *
            plane_spacing_mm
        )

        if area <= 0:
            continue

        # --------------------------------------------------------
        # Area-equivalent diameter
        # --------------------------------------------------------

        diameter = (
            2.0
            *
            np.sqrt(
                area / np.pi
            )
        )

        area_mm2[
            z, y, x
        ] = area

        diameter_mm[
            z, y, x
        ] = diameter

        valid[
            z, y, x
        ] = True

        # Optional progress

        if (
            point_number > 0
            and
            point_number % 1000 == 0
        ):

            print(
                f"  {point_number}/{n_points}"
            )

    # ============================================================
    # Convert outputs back to SimpleITK
    # ============================================================

    def array_to_image(
        array,
        reference,
    ):

        image = sitk.GetImageFromArray(
            array
        )

        image.CopyInformation(
            reference
        )

        return image

    diameter_img = array_to_image(
        diameter_mm,
        airway_img,
    )

    area_img = array_to_image(
        area_mm2,
        airway_img,
    )

    tangent_quality_img = array_to_image(
        tangent_quality,
        airway_img,
    )

    valid_img = array_to_image(
        valid.astype(np.uint8),
        airway_img,
    )

    return {
        "diameter_img":
            diameter_img,

        "area_img":
            area_img,

        "tangent_quality_img":
            tangent_quality_img,

        "valid_img":
            valid_img,

        "plane_spacing_mm":
            plane_spacing_mm,

        "tangent_radius_mm":
            tangent_radius_mm,
    }





def compute_airway_geometry_v0(
    airway_img,
    min_diameter_mm=2.0,
    max_diameter_mm=10.0,
):
    """
    Precompute airway geometry for the entire airway tree.

    Parameters
    ----------
    airway_img : sitk.Image
        Binary airway lumen segmentation.

    min_diameter_mm : float
        Minimum airway diameter to retain.

    max_diameter_mm : float
        Maximum airway diameter to retain.

    Returns
    -------
    dict containing:
        skeleton_img
        diameter_img
        valid_airway_img
        spacing_xyz
    """

    airway = sitk.GetArrayFromImage(airway_img) > 0

    # SITK: x,y,z
    # numpy: z,y,x
    spacing_xyz = np.asarray(
        airway_img.GetSpacing(),
        dtype=np.float64
    )

    spacing_zyx = spacing_xyz[::-1]

    # ---------------------------------------
    # Centerline approximation
    # ---------------------------------------
    print('skeleton')
    skeleton = skeletonize(airway)
    # ---------------------------------------
    # Physical distance transform
    # ---------------------------------------

    # radius_mm = distance_transform_edt(
    #     airway,
    #     sampling=spacing_zyx,
    # )

    print("Distance map")
    radius_img = sitk.SignedMaurerDistanceMap(
        sitk.Cast(airway_img > 0, sitk.sitkUInt8),
        insideIsPositive=True,
        squaredDistance=False,
        useImageSpacing=True,
    )

    radius_mm = sitk.GetArrayViewFromImage(
        radius_img
    )

    diameter_mm = np.zeros(
        airway.shape,
        dtype=np.float32
    )

    diameter_mm[skeleton>0] = (
        2.0 * radius_mm[skeleton>0]
    )

    # ---------------------------------------
    # Diameter selection
    # ---------------------------------------

    valid = (
        skeleton
        & (diameter_mm >= min_diameter_mm)
        & (diameter_mm <= max_diameter_mm)
    )

    # ---------------------------------------
    # Convert back to SITK
    # ---------------------------------------

    skeleton_img = sitk.GetImageFromArray(
        skeleton.astype(np.uint8)
    )
    skeleton_img.CopyInformation(airway_img)

    diameter_img = sitk.GetImageFromArray(
        diameter_mm
    )
    diameter_img.CopyInformation(airway_img)

    valid_img = sitk.GetImageFromArray(
        valid.astype(np.uint8)
    )
    valid_img.CopyInformation(airway_img)

    return {
        "skeleton_img": skeleton_img,
        "diameter_img": diameter_img,
        "radius_img": radius_img,
        "valid_airway_img": valid_img,
        "spacing_xyz": spacing_xyz,
        "min_diameter_mm": min_diameter_mm,
        "max_diameter_mm": max_diameter_mm,
    }


def compute_airway_geometry(
    airway_img,
    min_diameter_mm=2.0,
    max_diameter_mm=10.0,
    maximum_filter_size=3,
):
    """
    Precompute airway geometry for the entire airway tree.

    Airway diameter is estimated at each skeleton voxel as twice
    the local maximum of the Euclidean distance transform.

    Parameters
    ----------
    airway_img : sitk.Image
        Binary airway lumen segmentation.

    min_diameter_mm : float
        Minimum airway diameter to retain.

    max_diameter_mm : float
        Maximum airway diameter to retain.

    maximum_filter_size : int or tuple
        Neighborhood used to find the local maximum of the
        distance transform.

        Default = 3, corresponding to a 3x3x3 neighborhood.

    Returns
    -------
    dict containing:
        skeleton_img
        diameter_img
        diameter_raw_img
        valid_airway_img
        radius_img
        spacing_xyz
    """

    # --------------------------------------------------------
    # Binary airway
    # --------------------------------------------------------

    airway = (
        sitk.GetArrayFromImage(airway_img) > 0
    )

    # SITK: x,y,z
    # NumPy: z,y,x

    spacing_xyz = np.asarray(
        airway_img.GetSpacing(),
        dtype=np.float64,
    )

    spacing_zyx = spacing_xyz[::-1]

    # --------------------------------------------------------
    # Centerline approximation
    # --------------------------------------------------------

    print("Skeleton")

    skeleton = np.asarray(
        skeletonize(airway),
        dtype=bool,
    )

    skeleton_img = sitk.GetImageFromArray(
        skeleton.astype(np.uint8)
    )

    skeleton_img.CopyInformation(
        airway_img
    )

    # --------------------------------------------------------
    # Physical Euclidean distance transform
    # --------------------------------------------------------

    print("Distance map")

    # airway_binary_img = sitk.Cast(
    #     airway_img > 0,
    #     sitk.sitkUInt8,
    # )

    # radius_img = sitk.SignedMaurerDistanceMap(
    #     airway_binary_img,
    #     insideIsPositive=True,
    #     squaredDistance=False,
    #     useImageSpacing=True,
    # )

    # radius_mm = np.asarray(
    #     sitk.GetArrayViewFromImage(radius_img),
    #     dtype=np.float32,
    # )


    radius_mm = distance_transform_edt(
        airway,
        sampling=spacing_zyx).astype(np.float32)

    radius_img = sitk.GetImageFromArray(
        radius_mm.astype(np.float32)
    )

    radius_img.CopyInformation(
        airway_img
    )


    # --------------------------------------------------------
    # ORIGINAL diameter:
    #
    # D = 2 * EDT at skeleton voxel
    #
    # Keep this for QC/comparison.
    # --------------------------------------------------------

    diameter_raw_mm = np.zeros(
        airway.shape,
        dtype=np.float32,
    )

    diameter_raw_mm[skeleton] = (
        2.0 * radius_mm[skeleton]
    )

    # --------------------------------------------------------
    # LOCAL-MAXIMUM EDT
    #
    # Restrict distance map to airway lumen and find the
    # largest radius in the local neighborhood.
    # --------------------------------------------------------

    radius_inside_mm = np.where(
        airway,
        radius_mm,
        0.0,
    )

    radius_local_max_mm = maximum_filter(
        radius_inside_mm,
        size=maximum_filter_size,
        mode="constant",
        cval=0.0,
    )

    # --------------------------------------------------------
    # Refined diameter:
    #
    # D(c) = 2 * max EDT(x)
    #              x in neighborhood(c)
    # --------------------------------------------------------

    diameter_mm = np.zeros(
        airway.shape,
        dtype=np.float32,
    )

    diameter_mm[skeleton] = (
        2.0
        * radius_local_max_mm[skeleton]
    )


    #Compute Oblique angle size
    #-------------------------------------------------------
    # Airway exact diameter based on oblique angle resampling
    # -------------------------------------------------------

    print("Computing airway diameter based on oblique angle")
    oblique_geometry = compute_oblique_airway_diameters(
        airway_img=airway_img,
        skeleton_img=skeleton_img,
        edt_radius_img=radius_img,
        tangent_radius_mm=3.0,
        plane_spacing_mm=0.20,
        plane_margin_mm=2.0,
        min_half_width_mm=2.5,
        max_half_width_mm=7.0,
    )

    #Do final diameter assigment based on average between methods
    diameter_oblique_mm=sitk.GetArrayFromImage(oblique_geometry["diameter_img"])
    diameter_mm[skeleton] = 0.5*(diameter_mm[skeleton] + diameter_oblique_mm[skeleton])


    # --------------------------------------------------------
    # Diameter selection
    # --------------------------------------------------------

    valid = (
        skeleton
        & np.isfinite(diameter_mm)
        & (diameter_mm >= min_diameter_mm)
        & (diameter_mm <= max_diameter_mm)
    )

    # --------------------------------------------------------
    # Convert back to SimpleITK
    # --------------------------------------------------------

    diameter_img = sitk.GetImageFromArray(
        diameter_mm
    )

    diameter_img.CopyInformation(
        airway_img
    )

    diameter_oblique_img = sitk.GetImageFromArray(
        diameter_oblique_mm
    )

    diameter_oblique_img.CopyInformation(
        airway_img
    )

    # Original diameter image for QC

    diameter_raw_img = sitk.GetImageFromArray(
        diameter_raw_mm
    )

    diameter_raw_img.CopyInformation(
        airway_img
    )

    valid_img = sitk.GetImageFromArray(
        valid.astype(np.uint8)
    )

    valid_img.CopyInformation(
        airway_img
    )

    return {
        "skeleton_img":
            skeleton_img,
        # Refined local-maximum diameter
        "diameter_img":
            diameter_img,
        # Original skeleton-sampled EDT diameter
        "radius_img":
            radius_img,
        "diameter_oblique_img":
            diameter_oblique_img,
        "valid_airway_img":
            valid_img,
        "spacing_xyz":
            spacing_xyz,
        "min_diameter_mm":
            min_diameter_mm,
        "max_diameter_mm":
            max_diameter_mm,
        "maximum_filter_size":
            maximum_filter_size,
    }


def compute_region_alr(
    diameter_img,
    valid_airway_img,
    airway_region_img,
    volume_region_img=None,
    airway_img=None,
    return_diameters=False,
):
    """
    Compute ALR for an arbitrary binary region.

    ALR =
        geometric_mean(airway diameter)
        --------------------------------
                 region_volume^(1/3)

    If airway_img is provided, also calculate the actual segmented
    airway lumen volume within the region and DVI:

        DVI = (airway_volume / region_volume)^(1/3)

    Parameters
    ----------
    diameter_img : sitk.Image
        Precomputed centerline diameter image in mm.

    valid_airway_img : sitk.Image
        Binary centerline mask identifying valid diameter samples.

    airway_region_img : sitk.Image
        Binary expanded region used for airway assigment to include hilar region.

    volume_region_img : sitk.Image
        Binary anatomical region.

    airway_img : sitk.Image or None
        Original binary airway lumen segmentation. Required for
        airway-volume and DVI calculations.

    return_diameters : bool
        Return individual centerline diameter measurements.

    Returns
    -------
    dict
    """

    if volume_region_img is None:
        volume_region_img = airway_region_img

    diameter = sitk.GetArrayViewFromImage(
        diameter_img
    )

    valid_airway = (
        sitk.GetArrayViewFromImage(
            valid_airway_img
        ) > 0
    )

    # Expanded/Voronoi region:
    # determines which airways belong to this region
    airway_region = (
        sitk.GetArrayViewFromImage(
            airway_region_img
        ) > 0
    )

    # True anatomical region:
    # determines lung/lobar volume
    volume_region = (
        sitk.GetArrayViewFromImage(
            volume_region_img
        ) > 0
    )

    # -------------------------------------------------------
    # Geometry
    # -------------------------------------------------------

    spacing_xyz = np.asarray(
        volume_region_img.GetSpacing(),
        dtype=np.float64
    )

    voxel_volume_mm3 = float(
        np.prod(spacing_xyz)
    )

    # -------------------------------------------------------
    # Region volume
    # -------------------------------------------------------

    n_region_voxels = int(
        np.count_nonzero(volume_region)
    )

    if n_region_voxels == 0:
        return {
            "ALR": np.nan,
            "GM_diameter_mm": np.nan,
            "region_volume_ml": 0.0,
            "region_linear_size_mm": np.nan,
            "airway_volume_ml": np.nan,
            "DVI": np.nan,
            "n_centerline_samples": 0,
            "diameter_mean_mm": np.nan,
            "diameter_median_mm": np.nan,
            "diameter_std_mm": np.nan,
        }

    region_volume_mm3 = (
        n_region_voxels *
        voxel_volume_mm3
    )

    region_linear_size_mm = (
        region_volume_mm3 ** (1.0 / 3.0)
    )

    # -------------------------------------------------------
    # Actual airway lumen volume
    # -------------------------------------------------------

    airway_volume_mm3 = np.nan
    airway_volume_ml = np.nan
    dvi = np.nan

    if airway_img is not None:

        airway = (
            sitk.GetArrayViewFromImage(
                airway_img
            ) > 0
        )

        airway_in_region = (
            airway & airway_region
        )

        n_airway_voxels = int(
            np.count_nonzero(
                airway_in_region
            )
        )

        airway_volume_mm3 = (
            n_airway_voxels *
            voxel_volume_mm3
        )

        airway_volume_ml = (
            airway_volume_mm3 / 1000.0
        )

        dvi = (
            airway_volume_mm3 /
            region_volume_mm3
        ) ** (1.0 / 3.0)

    # -------------------------------------------------------
    # Select centerline diameter measurements
    # -------------------------------------------------------

    selection = (
        valid_airway
        & airway_region
    )

    diameters = np.asarray(
        diameter[selection],
        dtype=np.float64
    )

    diameters = diameters[
        np.isfinite(diameters)
        & (diameters > 0)
    ]

    # -------------------------------------------------------
    # No valid diameter samples
    # -------------------------------------------------------

    if diameters.size == 0:

        result = {
            "ALR": np.nan,
            "GM_diameter_mm": np.nan,
            "region_volume_ml":
                float(region_volume_mm3 / 1000.0),
            "region_linear_size_mm":
                float(region_linear_size_mm),
            "airway_volume_ml":
                float(airway_volume_ml),
            "DVI":
                float(dvi),
            "n_centerline_samples": 0,
            "diameter_mean_mm": np.nan,
            "diameter_median_mm": np.nan,
            "diameter_std_mm": np.nan,
        }

        if return_diameters:
            result["diameters_mm"] = diameters

        return result

    # -------------------------------------------------------
    # Geometric mean and ALR
    # -------------------------------------------------------

    gm_diameter_mm = float(
        np.exp(
            np.mean(
                np.log(diameters)
            )
        )
    )

    alr = (
        gm_diameter_mm /
        region_linear_size_mm
    )

    result = {
        "ALR":
            float(alr),

        "GM_diameter_mm":
            gm_diameter_mm,

        "region_volume_ml":
            float(region_volume_mm3 / 1000.0),

        "region_linear_size_mm":
            float(region_linear_size_mm),

        "airway_volume_ml":
            float(airway_volume_ml),

        "DVI":
            float(dvi),

        "n_centerline_samples":
            int(diameters.size),

        "diameter_mean_mm":
            float(np.mean(diameters)),

        "diameter_median_mm":
            float(np.median(diameters)),

        "diameter_std_mm":
            float(np.std(diameters)),
    }

    if return_diameters:
        result["diameters_mm"] = diameters

    return result


# ============================================================
# 26-connected neighborhood
# ============================================================

NEIGHBOR_OFFSETS_26 = [
    (dz, dy, dx)
    for dz in (-1, 0, 1)
    for dy in (-1, 0, 1)
    for dx in (-1, 0, 1)
    if not (dz == 0 and dy == 0 and dx == 0)
]


def compute_adaptive_diameter_groups(
    airway_geometry,
    min_diameter_mm=2.0,
    max_diameter_mm=10.0,

    # Initial region-growing criteria
    local_tolerance_mm=0.50,
    local_relative_tolerance=0.10,

    group_tolerance_mm=1.00,
    group_relative_tolerance=0.20,

    min_group_voxels=3,

    # Second-stage merging
    merge_adjacent_groups=True,
    target_n_groups=30,
    max_merge_difference_mm=1.25,
    max_merge_relative_difference=0.25,

    #Alternative diameter image with oblique computation to use for grouping
    measurement_diameter_img=None,
):
    """
    Create connected diameter-homogeneous airway groups.

    Stage 1
    -------
    Adaptive region growing using combined absolute and relative
    diameter tolerances.

    Stage 2
    -------
    Adjacent groups with similar representative diameters are
    iteratively merged.

    Merging stops when either:

        n_groups <= target_n_groups

    OR no adjacent pair satisfies the allowed diameter difference.

    Parameters
    ----------
    airway_geometry : dict
        Output of compute_airway_geometry().

    min_diameter_mm, max_diameter_mm : float
        Valid centerline diameter range.

    local_tolerance_mm : float
        Minimum absolute tolerance between neighboring voxels.

    local_relative_tolerance : float
        Relative tolerance between neighboring voxels.

    group_tolerance_mm : float
        Minimum absolute tolerance relative to group median.

    group_relative_tolerance : float
        Relative tolerance relative to group median.

    min_group_voxels : int
        Minimum group size retained.

    merge_adjacent_groups : bool
        Perform second-stage adjacency-constrained merging.

    target_n_groups : int or None
        Desired maximum number of groups.

        This is a soft target: groups will not be forced together
        if their diameters exceed the merge criteria.

    max_merge_difference_mm : float
        Maximum absolute difference in median diameter for merging.

    max_merge_relative_difference : float
        Maximum relative difference in median diameter for merging.

    measurement_diameter_img : SimpleITK Image
        Output distance image from compute_oblique_airway_geometry().

    Returns
    -------
    group_label_img : sitk.Image
        Integer group-label image.

    groups : list of dict
        Group measurements.
    """

    # ========================================================
    # Input
    # ========================================================

    skeleton_img = airway_geometry["skeleton_img"]
    diameter_img = airway_geometry["diameter_img"]

    skeleton = (
        sitk.GetArrayViewFromImage(skeleton_img) > 0
    )

    diameter = np.asarray(
        sitk.GetArrayViewFromImage(diameter_img),
        dtype=np.float32,
    )

    # Diameter used to MEASURE groups
    if measurement_diameter_img is None:
        measurement_diameter = diameter
    else:
        measurement_diameter = np.asarray(
            sitk.GetArrayViewFromImage(
                measurement_diameter_img
            ),
            dtype=np.float32,
        )

        if measurement_diameter.shape != diameter.shape:
            raise ValueError(
                "measurement_diameter_img must have the same "
                "geometry as diameter_img."
            )

    shape = skeleton.shape
    nz, ny, nx = shape

    spacing_xyz = np.asarray(
        skeleton_img.GetSpacing(),
        dtype=np.float64,
    )

    spacing_zyx = spacing_xyz[::-1]

    # ========================================================
    # Valid centerline
    # ========================================================

    valid = (
        skeleton
        & np.isfinite(diameter)
        & (diameter >= min_diameter_mm)
        & (diameter <= max_diameter_mm)
    )

    visited = np.zeros(
        shape,
        dtype=bool,
    )

    # Initial labels before merging
    initial_group_map = np.zeros(
        shape,
        dtype=np.int32,
    )

    initial_groups = {}

    next_group_id = 1

    # ========================================================
    # Helper: combined absolute + relative tolerance
    # ========================================================

    def diameter_similar(
        d1,
        d2,
        absolute_tolerance,
        relative_tolerance,
    ):

        reference = 0.5 * (
            abs(d1) + abs(d2)
        )

        tolerance = max(
            absolute_tolerance,
            relative_tolerance * reference,
        )

        return (
            abs(d1 - d2) <= tolerance
        )

    # ========================================================
    # STAGE 1
    # Adaptive region growing
    # ========================================================

    seed_coordinates = np.argwhere(
        valid
    )

    for seed in seed_coordinates:

        sz, sy, sx = map(
            int,
            seed,
        )

        if visited[
            sz, sy, sx
        ]:
            continue

        queue = deque([
            (sz, sy, sx)
        ])

        visited[
            sz, sy, sx
        ] = True

        group_coords = []
        group_values = []

        while queue:

            z, y, x = queue.popleft()

            current_diameter = float(
                diameter[z, y, x]
            )

            group_coords.append(
                (z, y, x)
            )

            group_values.append(
                current_diameter
            )

            # --------------------------------------------
            # Evolving robust group reference
            # --------------------------------------------

            group_reference = float(
                np.median(
                    group_values
                )
            )

            # --------------------------------------------
            # Examine 26-connected neighbors
            # --------------------------------------------

            for dz, dy, dx in NEIGHBOR_OFFSETS_26:

                zz = z + dz
                yy = y + dy
                xx = x + dx

                if (
                    zz < 0 or zz >= nz
                    or yy < 0 or yy >= ny
                    or xx < 0 or xx >= nx
                ):
                    continue

                if not valid[
                    zz, yy, xx
                ]:
                    continue

                if visited[
                    zz, yy, xx
                ]:
                    continue

                neighbor_diameter = float(
                    diameter[
                        zz, yy, xx
                    ]
                )

                # ----------------------------------------
                # Criterion 1:
                # local diameter continuity
                # ----------------------------------------

                if not diameter_similar(
                    current_diameter,
                    neighbor_diameter,
                    local_tolerance_mm,
                    local_relative_tolerance,
                ):
                    continue

                # ----------------------------------------
                # Criterion 2:
                # similarity to group median
                # ----------------------------------------

                if not diameter_similar(
                    group_reference,
                    neighbor_diameter,
                    group_tolerance_mm,
                    group_relative_tolerance,
                ):
                    continue

                # ----------------------------------------
                # Accept
                # ----------------------------------------

                visited[
                    zz, yy, xx
                ] = True

                queue.append(
                    (zz, yy, xx)
                )

        # ====================================================
        # Group complete
        # ====================================================

        if (
            len(group_coords)
            < min_group_voxels
        ):
            continue

        coords = np.asarray(
            group_coords,
            dtype=np.int32,
        )

        values = np.asarray(
            group_values,
            dtype=np.float64,
        )

        z = coords[:, 0]
        y = coords[:, 1]
        x = coords[:, 2]

        initial_group_map[
            z, y, x
        ] = next_group_id

        initial_groups[
            next_group_id
        ] = {
            "coords": coords,
            "values": values,
        }

        next_group_id += 1

    # ========================================================
    # STAGE 2
    # Merge adjacent groups
    # ========================================================

    if (
        merge_adjacent_groups
        and len(initial_groups) > 1
    ):

        # ----------------------------------------------------
        # Find which initial groups touch each other.
        #
        # This is done once from the original group map.
        # ----------------------------------------------------

        adjacency = {
            gid: set()
            for gid in initial_groups
        }

        occupied = (
            initial_group_map > 0
        )

        coords = np.argwhere(
            occupied
        )

        for z, y, x in coords:

            gid = int(
                initial_group_map[
                    z, y, x
                ]
            )

            for dz, dy, dx in NEIGHBOR_OFFSETS_26:

                zz = z + dz
                yy = y + dy
                xx = x + dx

                if (
                    zz < 0 or zz >= nz
                    or yy < 0 or yy >= ny
                    or xx < 0 or xx >= nx
                ):
                    continue

                other = int(
                    initial_group_map[
                        zz, yy, xx
                    ]
                )

                if (
                    other > 0
                    and other != gid
                ):
                    adjacency[
                        gid
                    ].add(other)

        # ----------------------------------------------------
        # Active group structure
        # ----------------------------------------------------

        active = {}

        for gid, data in initial_groups.items():

            active[gid] = {
                "members": {gid},

                "coords":
                    data["coords"],

                "values":
                    data["values"],

                "neighbors":
                    set(adjacency[gid]),
            }

        # ----------------------------------------------------
        # Representative diameter
        # ----------------------------------------------------

        def group_median(gid):

            return float(
                np.median(
                    active[gid][
                        "values"
                    ]
                )
            )

        # ----------------------------------------------------
        # Is a pair allowed to merge?
        # ----------------------------------------------------

        def merge_is_allowed(
            gid1,
            gid2,
        ):

            d1 = group_median(
                gid1
            )

            d2 = group_median(
                gid2
            )

            absolute_difference = abs(
                d1 - d2
            )

            reference = 0.5 * (
                d1 + d2
            )

            relative_difference = (
                absolute_difference
                / max(reference, 1e-6)
            )

            return (
                absolute_difference
                <= max_merge_difference_mm
                and
                relative_difference
                <= max_merge_relative_difference
            )

        # ----------------------------------------------------
        # Agglomerative merging
        # ----------------------------------------------------

        while True:

            n_active = len(
                active
            )

            # Target reached
            if (
                target_n_groups is not None
                and n_active
                <= target_n_groups
            ):
                break

            best_pair = None
            best_difference = np.inf

            # --------------------------------------------
            # Search only ADJACENT groups
            # --------------------------------------------

            for gid1 in list(
                active.keys()
            ):

                for gid2 in active[
                    gid1
                ]["neighbors"]:

                    if gid2 not in active:
                        continue

                    if gid2 <= gid1:
                        continue

                    if not merge_is_allowed(
                        gid1,
                        gid2,
                    ):
                        continue

                    difference = abs(
                        group_median(gid1)
                        -
                        group_median(gid2)
                    )

                    if (
                        difference
                        < best_difference
                    ):

                        best_difference = (
                            difference
                        )

                        best_pair = (
                            gid1,
                            gid2,
                        )

            # No anatomically/diametrically acceptable merge
            if best_pair is None:
                break

            gid1, gid2 = best_pair

            # --------------------------------------------
            # Merge gid2 into gid1
            # --------------------------------------------

            active[gid1][
                "coords"
            ] = np.vstack([
                active[gid1]["coords"],
                active[gid2]["coords"],
            ])

            active[gid1][
                "values"
            ] = np.concatenate([
                active[gid1]["values"],
                active[gid2]["values"],
            ])

            active[gid1][
                "members"
            ].update(
                active[gid2][
                    "members"
                ]
            )

            # --------------------------------------------
            # Update adjacency
            # --------------------------------------------

            new_neighbors = (
                active[gid1][
                    "neighbors"
                ]
                |
                active[gid2][
                    "neighbors"
                ]
            )

            new_neighbors.discard(
                gid1
            )

            new_neighbors.discard(
                gid2
            )

            active[gid1][
                "neighbors"
            ] = new_neighbors

            # Update neighbors pointing to gid2
            for neighbor in list(
                new_neighbors
            ):

                if neighbor not in active:
                    continue

                active[neighbor][
                    "neighbors"
                ].discard(
                    gid2
                )

                active[neighbor][
                    "neighbors"
                ].add(
                    gid1
                )

            del active[
                gid2
            ]

    else:

        # No merging
        active = {}

        for gid, data in initial_groups.items():

            active[gid] = {
                "members": {gid},
                "coords": data["coords"],
                "values": data["values"],
                "neighbors": set(),
            }

    # ========================================================
    # Rebuild final group map and measurements
    # ========================================================

    group_map = np.zeros(
        shape,
        dtype=np.int32,
    )

    groups = []

    new_group_id = 1

    for _, data in active.items():

        coords = data[
            "coords"
        ]

        values = np.asarray(
            data["values"],
            dtype=np.float64,
        )

        z = coords[:, 0]
        y = coords[:, 1]
        x = coords[:, 2]

        group_map[
            z, y, x
        ] = new_group_id

        # ----------------------------------------------------
        # Diameter measurements
        # ----------------------------------------------------

        median_diameter = float(
            np.median(values)
        )

        mean_diameter = float(
            np.mean(values)
        )

        gm_diameter = float(
            np.exp(
                np.mean(
                    np.log(values)
                )
            )
        )

        # ----------------------------------------------------
        # Physical extent
        # ----------------------------------------------------

        physical_coords = (
            coords
            * spacing_zyx
        )

        extent_mm = np.ptp(
            physical_coords,
            axis=0,
        )

        # ----------------------------------------------------
        # Save
        # ----------------------------------------------------

        groups.append(
            {
                "group_id":
                    new_group_id,

                "median_diameter_mm":
                    median_diameter,

                "mean_diameter_mm":
                    mean_diameter,

                "gm_diameter_mm":
                    gm_diameter,

                "diameter_std_mm":
                    float(
                        np.std(values)
                    ),

                "diameter_min_mm":
                    float(
                        np.min(values)
                    ),

                "diameter_max_mm":
                    float(
                        np.max(values)
                    ),

                "n_centerline_voxels":
                    int(
                        len(values)
                    ),

                "extent_z_mm":
                    float(
                        extent_mm[0]
                    ),

                "extent_y_mm":
                    float(
                        extent_mm[1]
                    ),

                "extent_x_mm":
                    float(
                        extent_mm[2]
                    ),

                # Useful QC:
                # how many original groups were combined?
                "n_initial_groups":
                    int(
                        len(
                            data["members"]
                        )
                    ),
            }
        )

        new_group_id += 1

    # ========================================================
    # SimpleITK output
    # ========================================================

    group_label_img = sitk.GetImageFromArray(
        group_map
    )

    group_label_img.CopyInformation(
        skeleton_img
    )

    return group_label_img, groups



def compute_adaptive_diameter_groups_v0(
    airway_geometry,
    min_diameter_mm=2.0,
    max_diameter_mm=10.0,
    local_tolerance_mm=0.35,
    group_tolerance_mm=0.75,
    min_group_voxels=3,
):
    """
    Create spatially connected, diameter-homogeneous airway groups
    from a precomputed airway skeleton and diameter map.

    This is intended as a simplified alternative to explicit airway
    branch extraction.

    Parameters
    ----------
    airway_geometry : dict
        Output of compute_airway_geometry(), containing at least:

            airway_geometry["skeleton_img"]
            airway_geometry["diameter_img"]

    min_diameter_mm : float
        Minimum centerline diameter considered.

    max_diameter_mm : float
        Maximum centerline diameter considered.

    local_tolerance_mm : float
        Maximum allowed diameter difference between two neighboring
        skeleton voxels:

            abs(D_neighbor - D_current) <= local_tolerance_mm

        This allows smooth changes in caliber along an airway.

    group_tolerance_mm : float
        Maximum allowed difference between a candidate voxel and the
        current representative diameter of the group:

            abs(D_neighbor - median(D_group)) <= group_tolerance_mm

        This prevents cumulative diameter drift.

    min_group_voxels : int
        Minimum number of centerline voxels required for a group to
        be retained.

    Returns
    -------
    group_label_img : sitk.Image
        Integer label image.

        0 = no valid group
        1...N = adaptive diameter groups

    groups : list of dict
        Measurements for each retained group.
    """

    # --------------------------------------------------------
    # Get skeleton and diameter images
    # --------------------------------------------------------

    skeleton_img = airway_geometry["skeleton_img"]
    diameter_img = airway_geometry["diameter_img"]

    skeleton = (
        sitk.GetArrayFromImage(skeleton_img) > 0
    )

    diameter = np.asarray(
        sitk.GetArrayFromImage(diameter_img),
        dtype=np.float32,
    )

    shape = skeleton.shape
    nz, ny, nx = shape

    # --------------------------------------------------------
    # Only consider valid diameter range
    # --------------------------------------------------------

    valid = (
        skeleton
        & np.isfinite(diameter)
        & (diameter >= min_diameter_mm)
        & (diameter <= max_diameter_mm)
    )

    # --------------------------------------------------------
    # State arrays
    # --------------------------------------------------------

    visited = np.zeros(
        shape,
        dtype=bool,
    )

    group_map = np.zeros(
        shape,
        dtype=np.int32,
    )

    groups = []

    next_group_id = 1

    # --------------------------------------------------------
    # Iterate through valid skeleton voxels
    # --------------------------------------------------------

    seed_coordinates = np.argwhere(valid)

    for seed in seed_coordinates:

        sz, sy, sx = seed

        if visited[sz, sy, sx]:
            continue

        # ----------------------------------------------------
        # Start new adaptive group
        # ----------------------------------------------------

        queue = deque()

        queue.append(
            (int(sz), int(sy), int(sx))
        )

        visited[sz, sy, sx] = True

        group_coords = []
        group_values = []

        # ----------------------------------------------------
        # Region growing
        # ----------------------------------------------------

        while queue:

            z, y, x = queue.popleft()

            current_diameter = float(
                diameter[z, y, x]
            )

            group_coords.append(
                (z, y, x)
            )

            group_values.append(
                current_diameter
            )

            # ------------------------------------------------
            # Robust evolving group reference
            # ------------------------------------------------

            group_reference = float(
                np.median(group_values)
            )

            # ------------------------------------------------
            # Examine connected skeleton neighbors
            # ------------------------------------------------

            for dz, dy, dx in NEIGHBOR_OFFSETS_26:

                zz = z + dz
                yy = y + dy
                xx = x + dx

                if (
                    zz < 0 or zz >= nz
                    or yy < 0 or yy >= ny
                    or xx < 0 or xx >= nx
                ):
                    continue

                if not valid[zz, yy, xx]:
                    continue

                if visited[zz, yy, xx]:
                    continue

                neighbor_diameter = float(
                    diameter[zz, yy, xx]
                )

                # --------------------------------------------
                # Criterion 1:
                # local continuity
                # --------------------------------------------

                local_difference = abs(
                    neighbor_diameter
                    - current_diameter
                )

                if (
                    local_difference
                    > local_tolerance_mm
                ):
                    continue

                # --------------------------------------------
                # Criterion 2:
                # similarity to the overall group
                # --------------------------------------------

                group_difference = abs(
                    neighbor_diameter
                    - group_reference
                )

                if (
                    group_difference
                    > group_tolerance_mm
                ):
                    continue

                # --------------------------------------------
                # Accept
                # --------------------------------------------

                visited[zz, yy, xx] = True

                queue.append(
                    (zz, yy, xx)
                )

        # ----------------------------------------------------
        # Group complete
        # ----------------------------------------------------

        group_values = np.asarray(
            group_values,
            dtype=np.float64,
        )

        n_voxels = len(group_values)

        # ----------------------------------------------------
        # Ignore tiny groups
        # ----------------------------------------------------

        if n_voxels < min_group_voxels:
            continue

        # ----------------------------------------------------
        # Representative diameter
        # ----------------------------------------------------

        median_diameter = float(
            np.median(group_values)
        )

        mean_diameter = float(
            np.mean(group_values)
        )

        gm_diameter = float(
            np.exp(
                np.mean(
                    np.log(group_values)
                )
            )
        )

        # ----------------------------------------------------
        # Assign group ID
        # ----------------------------------------------------

        coords_array = np.asarray(
            group_coords,
            dtype=np.int32,
        )

        z = coords_array[:, 0]
        y = coords_array[:, 1]
        x = coords_array[:, 2]

        group_map[
            z, y, x
        ] = next_group_id

        # ----------------------------------------------------
        # Physical extent
        #
        # This is not true branch length. It is simply useful
        # descriptive information.
        # ----------------------------------------------------

        spacing_xyz = np.asarray(
            skeleton_img.GetSpacing(),
            dtype=np.float64,
        )

        spacing_zyx = spacing_xyz[::-1]

        physical_coords = (
            coords_array
            * spacing_zyx
        )

        extent_mm = np.ptp(
            physical_coords,
            axis=0,
        )

        groups.append(
            {
                "group_id":
                    next_group_id,

                "median_diameter_mm":
                    median_diameter,

                "mean_diameter_mm":
                    mean_diameter,

                "gm_diameter_mm":
                    gm_diameter,

                "diameter_std_mm":
                    float(
                        np.std(group_values)
                    ),

                "diameter_min_mm":
                    float(
                        np.min(group_values)
                    ),

                "diameter_max_mm":
                    float(
                        np.max(group_values)
                    ),

                "n_centerline_voxels":
                    int(n_voxels),

                "extent_z_mm":
                    float(extent_mm[0]),

                "extent_y_mm":
                    float(extent_mm[1]),

                "extent_x_mm":
                    float(extent_mm[2]),
            }
        )

        next_group_id += 1

    # --------------------------------------------------------
    # Convert labels to SimpleITK
    # --------------------------------------------------------

    group_label_img = sitk.GetImageFromArray(
        group_map
    )

    group_label_img.CopyInformation(
        skeleton_img
    )

    return group_label_img, groups



def compute_region_alr_from_adaptive_groups(
    group_label_img,
    groups,
    region_img,
    volume_region_img=None,
    min_region_fraction=0.5,
    representative_diameter="median",
):
    """
    Compute regional ALR from adaptive diameter groups.

    Each adaptive airway group contributes exactly one diameter
    to the final geometric mean.
    """

    if volume_region_img is None:
        volume_region_img = region_img

    group_map = sitk.GetArrayViewFromImage(
        group_label_img
    )

    region = (
        sitk.GetArrayViewFromImage(
            region_img
        ) > 0
    )

    volume_region = (
        sitk.GetArrayViewFromImage(
            volume_region_img
        ) > 0
    )

    diameter_keys = {
        "median": "median_diameter_mm",
        "mean": "mean_diameter_mm",
        "gm": "gm_diameter_mm",
    }

    if representative_diameter not in diameter_keys:
        raise ValueError(
            "representative_diameter must be "
            "'median', 'mean', or 'gm'."
        )

    diameter_key = diameter_keys[
        representative_diameter
    ]

    # --------------------------------------------------------
    # Anatomical region volume
    # --------------------------------------------------------

    spacing_xyz = np.asarray(
        volume_region_img.GetSpacing(),
        dtype=np.float64,
    )

    voxel_volume_mm3 = np.prod(
        spacing_xyz
    )

    region_volume_mm3 = (
        np.count_nonzero(volume_region)
        * voxel_volume_mm3
    )

    if region_volume_mm3 == 0:

        return {
            "ALR": np.nan,
            "GM_group_diameter_mm": np.nan,
            "n_groups": 0,
            "region_volume_ml": 0.0,
        }

    # --------------------------------------------------------
    # Select groups belonging to region
    # --------------------------------------------------------

    selected_diameters = []

    for group in groups:

        group_id = group["group_id"]

        group_voxels = (
            group_map == group_id
        )

        n_total = np.count_nonzero(
            group_voxels
        )

        if n_total == 0:
            continue

        n_inside = np.count_nonzero(
            group_voxels & region
        )

        region_fraction = (
            n_inside / n_total
        )

        if (
            region_fraction
            < min_region_fraction
        ):
            continue

        selected_diameters.append(
            group[diameter_key]
        )

    # --------------------------------------------------------
    # ALR
    # --------------------------------------------------------

    if len(selected_diameters) == 0:

        return {
            "ALR": np.nan,
            "GM_group_diameter_mm": np.nan,
            "n_groups": 0,
            "region_volume_ml":
                float(region_volume_mm3 / 1000.0),
        }

    selected_diameters = np.asarray(
        selected_diameters,
        dtype=np.float64,
    )

    gm_diameter = float(
        np.exp(
            np.mean(
                np.log(
                    selected_diameters
                )
            )
        )
    )

    region_linear_size_mm = (
        region_volume_mm3 ** (1.0 / 3.0)
    )

    alr = (
        gm_diameter
        / region_linear_size_mm
    )

    return {
        "ALR":
            float(alr),

        "GM_group_diameter_mm":
            gm_diameter,

        "n_groups":
            int(
                len(selected_diameters)
            ),

        "region_volume_ml":
            float(
                region_volume_mm3 / 1000.0
            ),

        "region_linear_size_mm":
            float(region_linear_size_mm),

        "group_diameter_mean_mm":
            float(
                np.mean(selected_diameters)
            ),

        "group_diameter_median_mm":
            float(
                np.median(selected_diameters)
            ),

        "group_diameter_std_mm":
            float(
                np.std(selected_diameters)
            ),
    }


## EXample of use 
# # ------------------------------------------------------------
# # Once per CT
# # ------------------------------------------------------------

# geometry = compute_airway_geometry(
#     airway_img,
#     min_diameter_mm=2,
#     max_diameter_mm=10,
# )

# group_img, groups = compute_adaptive_diameter_groups(
#     geometry,
#     min_diameter_mm=2,
#     max_diameter_mm=10,
#     local_tolerance_mm=0.35,
#     group_tolerance_mm=0.75,
#     min_group_voxels=3,
# )


# ## Test images
# sitk.WriteImage(
#     geometry["skeleton_img"],
#     "airway_skeleton.nii.gz"
# )

# sitk.WriteImage(
#     geometry["diameter_img"],
#     "airway_diameter_mm.nii.gz"
# )

# sitk.WriteImage(
#     geometry["valid_airway_img"],
#     "airway_valid_2to10mm.nii.gz"
# )


# # ------------------------------------------------------------
# # Repeated for any region
# # ------------------------------------------------------------


# #Simple ALR approach
# whole_result = compute_region_alr(
#     diameter_img,
#     valid_airway_img,
#     whole_lung_mask,
# )

# right_result = compute_region_alr(
#     diameter_img,
#     valid_airway_img,
#     right_lung_mask,
# )

# #Approach with adapting groups


# whole = compute_region_alr_from_adaptive_groups(
#     group_img,
#     groups,
#     region_img=whole_lung_mask,
# )

# right = compute_region_alr_from_adaptive_groups(
#     group_img,
#     groups,
#     region_img=right_lung_mask,
# )


def restrict_airway_to_lung_z_extent(
    airway_img,
    region_label_img,
    lung_labels=(4, 5, 6, 7, 8),
    z_margin_slices=0,
):
    """
    Restrict airway segmentation to the superior-inferior (z)
    extent of the lung fields.

    The airway is NOT restricted in x/y. Therefore tracheal and
    extraparenchymal/hilar airway voxels are preserved as long as
    they lie between the most superior and inferior lung slices.

    Parameters
    ----------
    airway_img : sitk.Image
        Binary airway segmentation.

    region_label_img : sitk.Image
        Lung/lobar label map.

    lung_labels : sequence of int
        Labels defining the lung fields.

    z_margin_slices : int
        Optional number of slices to retain above and below the
        lung extent. Default = 0.

    Returns
    -------
    sitk.Image
        Airway image restricted to lung z extent.
    """

    airway = (
        sitk.GetArrayFromImage(airway_img) > 0
    )

    regions = sitk.GetArrayViewFromImage(
        region_label_img
    )

    lung = np.isin(
        regions,
        lung_labels,
    )

    # Find slices containing lung
    lung_slices = np.where(
        np.any(lung, axis=(1, 2))
    )[0]

    if len(lung_slices) == 0:
        raise ValueError(
            "No lung voxels found in region_label_img."
        )

    z_min = max(
        0,
        int(lung_slices[0]) - z_margin_slices,
    )

    z_max = min(
        airway.shape[0] - 1,
        int(lung_slices[-1]) + z_margin_slices,
    )

    # Restrict airway in z only
    restricted = np.zeros_like(
        airway,
        dtype=np.uint8,
    )

    restricted[
        z_min:z_max + 1
    ] = airway[
        z_min:z_max + 1
    ]

    out = sitk.GetImageFromArray(
        restricted
    )

    out.CopyInformation(
        airway_img
    )

    return out


def build_arg_parser():

    parser = argparse.ArgumentParser(
        description=(
            "Compute CT airway dysanapsis phenotypes "
            "from an airway segmentation and anatomical "
            "region label map."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # -------------------------------------------------------
    # Required inputs
    # -------------------------------------------------------

    parser.add_argument(
        "--airway",
        required=True,
        help=(
            "Binary airway lumen segmentation "
            "(NIfTI, NRRD, MHA, etc.)."
        ),
    )

    parser.add_argument(
        "--regions",
        required=True,
        help=(
            "Integer anatomical region label map. "
            "For example, a lobar segmentation."
        ),
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output CSV filename.",
    )

    # -------------------------------------------------------
    # Region definitions
    # -------------------------------------------------------

    parser.add_argument(
        "--region",
        action="append",
        type=parse_region_definition,
        help=(
            "Region definition NAME:label[,label,...]. "
            "May be specified multiple times. "
            "Example: --region RUL:1 "
            "--region RIGHT:1,2,3"
        ),
    )

    # -------------------------------------------------------
    # Airway diameter range
    # -------------------------------------------------------

    parser.add_argument(
        "--min-diameter",
        type=float,
        default=2.0,
        help="Minimum airway diameter in mm.",
    )

    parser.add_argument(
        "--max-diameter",
        type=float,
        default=10.0,
        help="Maximum airway diameter in mm.",
    )

    # -------------------------------------------------------
    # Adaptive-group parameters
    # -------------------------------------------------------

    parser.add_argument(
        "--local-tolerance",
        type=float,
        default=0.5,
        help=(
            "Maximum local diameter difference in mm "
            "for adaptive group growing."
        ),
    )

    parser.add_argument(
        "--group-tolerance",
        type=float,
        default=1.0,
        help=(
            "Maximum difference from adaptive-group "
            "representative diameter in mm."
        ),
    )

    parser.add_argument(
        "--min-group-voxels",
        type=int,
        default=3,
        help=(
            "Minimum number of centerline voxels "
            "in an adaptive diameter group."
        ),
    )

    parser.add_argument(
        "--min-region-fraction",
        type=float,
        default=0.5,
        help=(
            "Minimum fraction of an adaptive group "
            "that must lie inside a region."
        ),
    )

    parser.add_argument(
        "--representative-diameter",
        choices=[
            "median",
            "mean",
            "gm",
        ],
        default="median",
        help=(
            "Representative diameter assigned to "
            "each adaptive airway group."
        ),
    )

    parser.add_argument(
        "--voronoi-distance",
        type=float,
        default=20.0,
        help=(
            "Maximum distance in mm outside the true "
            "anatomical region for Voronoi airway assignment."
        ),
    )

    # -------------------------------------------------------
    # QC outputs
    # -------------------------------------------------------

    parser.add_argument(
        "--write-intermediates",
        default=None,
        metavar="DIRECTORY",
        help=(
            "Optional directory for QC images: "
            "skeleton, diameter map, valid-airway mask, "
            "and adaptive-group labels."
        ),
    )

    return parser



def run_analysis(args):

    # -------------------------------------------------------
    # Read images
    # -------------------------------------------------------

    print(f"Reading airway segmentation: {args.airway}")

    airway_img = sitk.ReadImage(
        args.airway
    )

    print(f"Reading region map: {args.regions}")

    region_label_img = sitk.ReadImage(
        args.regions
    )

    check_same_geometry(
        airway_img,
        region_label_img,
        "airway",
        "regions",
    )

    # -------------------------------------------------------
    # Basic parameter validation
    # -------------------------------------------------------

    if args.min_diameter <= 0:
        raise ValueError(
            "--min-diameter must be > 0."
        )

    if args.max_diameter <= args.min_diameter:
        raise ValueError(
            "--max-diameter must be greater "
            "than --min-diameter."
        )

    if not (
        0.0 <= args.min_region_fraction <= 1.0
    ):
        raise ValueError(
            "--min-region-fraction must be "
            "between 0 and 1."
        )

    # -------------------------------------------------------
    # Regions
    # -------------------------------------------------------

    if args.region:

        region_definitions = args.region

    else:

        # Default lobar convention.
        #
        # Change these if your label map follows
        # a different convention.

        region_definitions = [
            ("WHOLE", [4, 5, 6, 7, 8]),
            ("RIGHT", [4, 5, 6]),
            ("LEFT", [7, 8]),
            ("RUL", [4]),
            ("RML", [5]),
            ("RLL", [6]),
            ("LUL", [7]),
            ("LLL", [8]),
        ]

        print(
            "No --region arguments supplied. "
            "Using default labels:"
        )

        for name, labels in region_definitions:
            print(
                f"  {name}: {labels}"
            )

    # -------------------------------------------------------
    # Crop airway image to the lung zone along the z axis
    # -------------------------------------------------------

    airway_img = restrict_airway_to_lung_z_extent(
        airway_img=airway_img,
        region_label_img=region_label_img,
        lung_labels=dict(region_definitions)["WHOLE"]
    )

    # -------------------------------------------------------
    # Airway geometry
    # -------------------------------------------------------

    print("Computing airway skeleton and diameter map...")

    geometry = compute_airway_geometry(
        airway_img,
        min_diameter_mm=args.min_diameter,
        max_diameter_mm=args.max_diameter,
    )


    # -------------------------------------------------------
    # Adaptive groups
    # -------------------------------------------------------

    print("Computing adaptive diameter groups...")

    group_img, groups = (
        compute_adaptive_diameter_groups(
            geometry,
            min_diameter_mm=args.min_diameter,
            max_diameter_mm=args.max_diameter,
            local_tolerance_mm=args.local_tolerance,
            group_tolerance_mm=args.group_tolerance,
            min_group_voxels=args.min_group_voxels,
        )
    )

    print(
        f"Retained adaptive groups: {len(groups)}"
    )


    # print("Computing Voronoi-expanded regional map...")
    voronoi_region_img = None
    # voronoi_region_img = compute_voronoi_region_map(
    #     region_label_img,
    #     support_mask_img=None,
    #     background_label=0,
    #     max_assignment_distance_mm=args.voronoi_distance,
    # )


    print ("Label extraparenchymal airways....")

    external_label_img, external_info = (
        compute_external_airway_labels(
            geometry["skeleton_img"],
            region_label_img,
        )
    )

    # -------------------------------------------------------
    # Optional intermediate images
    # -------------------------------------------------------

    if args.write_intermediates:

        os.makedirs(
            args.write_intermediates,
            exist_ok=True,
        )

        sitk.WriteImage(
            geometry["skeleton_img"],
            os.path.join(
                args.write_intermediates,
                "airway_skeleton.nii.gz",
            ),
        )

        sitk.WriteImage(
            geometry["diameter_img"],
            os.path.join(
                args.write_intermediates,
                "airway_diameter_mm.nii.gz",
            ),
        )

        sitk.WriteImage(
            geometry["diameter_oblique_img"],
            os.path.join(
                args.write_intermediates,
                "airway_diameter_oblique_mm.nii.gz",
            ),
        )

        sitk.WriteImage(
            geometry["valid_airway_img"],
            os.path.join(
                args.write_intermediates,
                "airway_valid.nii.gz",
            ),
        )

        sitk.WriteImage(
            group_img,
            os.path.join(
                args.write_intermediates,
                "airway_adaptive_groups.nii.gz",
            ),
        )

        if voronoi_region_img is not None:
            sitk.WriteImage(
                voronoi_region_img,
                os.path.join(
                    args.write_intermediates,
                    "voronoi_region_map.nii.gz",
                ),
            )

        sitk.WriteImage(
            external_label_img,
            os.path.join(
                args.write_intermediates,
                "external_labels.nii.gz",
            ),
        )

    # -------------------------------------------------------
    # Regional analysis
    # -------------------------------------------------------

    rows = []

    for region_name, labels in region_definitions:

        print(
            f"Computing {region_name}: labels={labels}"
        )

        # True anatomical region
        region_img = make_binary_region(
            region_label_img,
            labels,
        )

        # Expanded region using Voronoi for airway assignment
        # airway_region_img = make_binary_region(
        #     voronoi_region_img,
        #     labels,
        # )


        #Using extarn label assigment to include extrapulmonary airways based on region
        airway_region_img = make_valid_airway_region(
            airway_img=airway_img,
            region_img=region_img,
            external_label_img=external_label_img,
            external_labels=EXTERNAL_REGION_LABELS[
                region_name
            ],
        )

        # -----------------------------------------------
        # Standard centerline ALR
        # -----------------------------------------------

        standard = compute_region_alr(
            diameter_img=geometry["diameter_img"],
            valid_airway_img=geometry[
                "valid_airway_img"
            ],
            airway_region_img=airway_region_img,
            volume_region_img=region_img,
            airway_img=airway_img,
        )

        # -----------------------------------------------
        # Adaptive-group ALR
        # -----------------------------------------------
        adaptive = (
            compute_region_alr_from_adaptive_groups(
                group_label_img=group_img,
                groups=groups,
                # Expanded region for airway assignment
                region_img=airway_region_img,
                # True region for volume
                volume_region_img=region_img,
                min_region_fraction=
                    args.min_region_fraction,
                representative_diameter=
                    args.representative_diameter,
            )
        )

        # -----------------------------------------------
        # Combined output row
        # -----------------------------------------------

        row = {
            "region":
                region_name,

            "labels":
                ",".join(
                    str(x)
                    for x in labels
                ),

            # Anatomy
            "region_volume_ml":
                standard["region_volume_ml"],

            "airway_volume_ml":
                standard["airway_volume_ml"],

            "DVI":
                standard["DVI"],

            # Standard ALR
            "ALR":
                standard["ALR"],

            "GM_diameter_mm":
                standard["GM_diameter_mm"],

            "n_centerline_samples":
                standard[
                    "n_centerline_samples"
                ],

            "diameter_mean_mm":
                standard[
                    "diameter_mean_mm"
                ],

            "diameter_median_mm":
                standard[
                    "diameter_median_mm"
                ],

            "diameter_std_mm":
                standard[
                    "diameter_std_mm"
                ],

            # Adaptive ALR
            "adaptive_ALR":
                adaptive["ALR"],

            "adaptive_GM_diameter_mm":
                adaptive[
                    "GM_group_diameter_mm"
                ],

            "n_adaptive_groups":
                adaptive["n_groups"],

            "adaptive_diameter_mean_mm":
                adaptive.get(
                    "group_diameter_mean_mm",
                    np.nan,
                ),

            "adaptive_diameter_median_mm":
                adaptive.get(
                    "group_diameter_median_mm",
                    np.nan,
                ),

            "adaptive_diameter_std_mm":
                adaptive.get(
                    "group_diameter_std_mm",
                    np.nan,
                ),
        }

        rows.append(row)

    # -------------------------------------------------------
    # Write CSV
    # -------------------------------------------------------

    output_dir = os.path.dirname(
        os.path.abspath(args.output)
    )

    if output_dir:
        os.makedirs(
            output_dir,
            exist_ok=True,
        )

    fieldnames = list(
        rows[0].keys()
    )

    with open(
        args.output,
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            rows
        )

    print("")
    print(
        f"Results written to: {args.output}"
    )

    return rows



def main():

    parser = build_arg_parser()

    args = parser.parse_args()

    try:

        run_analysis(args)

    except Exception as exc:

        print(
            f"ERROR: {exc}",
            file=sys.stderr,
        )

        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())



##Usage: python airway_dysanapsis_phenotypes.py \
##    --airway airway.nii.gz \
##    --regions lobes.nii.gz \
##    --output dysanapsis.csv \
##    --write-intermediates qc


