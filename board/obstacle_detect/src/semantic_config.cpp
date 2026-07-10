#include "../include/semantic_config.hpp"

#include <algorithm>

#ifndef A1_YOLO_NUM_CLASSES
#define A1_YOLO_NUM_CLASSES 80
#endif

namespace obstacle {
namespace semantic {
namespace {

const char* const kCocoNames[] = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
    "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
};

const char* const kRod25Names[] = {
    "bike",
    "building",
    "car",
    "person",
    "stairs",
    "traffic_sign",
    "electrical_pole",
    "road",
    "motorcycle",
    "dustbin",
    "dog",
    "manhole",
    "tree",
    "guard_rail",
    "pedestrian_crosswalk",
    "truck",
    "bus",
    "bench",
    "traffic_cone",
    "fire_hydrant",
    "teraffic_barrel",
    "plant_pot",
    "electrical_box",
    "chair",
    "bicycle_rack"
};

const int kRod25SemanticMap[] = {
    VEHICLE_BICYCLE,   // bike
    GENERIC_OBSTACLE,  // building
    VEHICLE_BICYCLE,   // car
    PERSON,            // person
    GENERIC_OBSTACLE,  // stairs
    GENERIC_OBSTACLE,  // traffic_sign
    GENERIC_OBSTACLE,  // electrical_pole
    GENERIC_OBSTACLE,  // road, filtered by IsSupportedRawClass
    VEHICLE_BICYCLE,   // motorcycle
    GENERIC_OBSTACLE,  // dustbin
    GENERIC_OBSTACLE,  // dog
    GENERIC_OBSTACLE,  // manhole
    GENERIC_OBSTACLE,  // tree
    GENERIC_OBSTACLE,  // guard_rail
    GENERIC_OBSTACLE,  // pedestrian_crosswalk
    VEHICLE_BICYCLE,   // truck
    VEHICLE_BICYCLE,   // bus
    CHAIR_SEAT,        // bench
    GENERIC_OBSTACLE,  // traffic_cone
    GENERIC_OBSTACLE,  // fire_hydrant
    GENERIC_OBSTACLE,  // teraffic_barrel
    SMALL_OBJECT,      // plant_pot
    GENERIC_OBSTACLE,  // electrical_box
    CHAIR_SEAT,        // chair
    GENERIC_OBSTACLE   // bicycle_rack
};

const char* const kSemanticNames[] = {
    "person",
    "chair/seat",
    "table/desk",
    "sofa/bed",
    "bag/suitcase",
    "small_object",
    "vehicle/bicycle",
    "generic_obstacle"
};

const char* const kSemanticShortNames[] = {
    "PER",
    "SEAT",
    "TAB",
    "SOFA",
    "BAG",
    "SMALL",
    "VEH",
    "OBS"
};

bool any_of(int value, const int* values, int count)
{
    for (int i = 0; i < count; ++i) {
        if (value == values[i]) return true;
    }
    return false;
}

}  // namespace

int ModelClassCount()
{
    return A1_YOLO_NUM_CLASSES;
}

bool IsSupportedRawClass(int raw_class_id)
{
    if (raw_class_id < 0 || raw_class_id >= ModelClassCount()) {
        return false;
    }
    if (ModelClassCount() == 25 && raw_class_id == 7) {
        return false;
    }
    return true;
}

int SemanticClassFromRaw(int raw_class_id)
{
    if (ModelClassCount() == NUM_SEMANTIC_CLASSES) {
        return IsSupportedRawClass(raw_class_id) ? raw_class_id : GENERIC_OBSTACLE;
    }

    if (ModelClassCount() == 25) {
        const int n = static_cast<int>(sizeof(kRod25SemanticMap) / sizeof(kRod25SemanticMap[0]));
        if (raw_class_id >= 0 && raw_class_id < n) {
            return kRod25SemanticMap[raw_class_id];
        }
        return GENERIC_OBSTACLE;
    }

    if (raw_class_id == 0) return PERSON;

    const int chair_seat[] = {13, 56};
    if (any_of(raw_class_id, chair_seat, 2)) return CHAIR_SEAT;

    const int table_desk[] = {60};
    if (any_of(raw_class_id, table_desk, 1)) return TABLE_DESK;

    const int sofa_bed[] = {57, 59};
    if (any_of(raw_class_id, sofa_bed, 2)) return SOFA_BED;

    const int bag_suitcase[] = {24, 26, 28};
    if (any_of(raw_class_id, bag_suitcase, 3)) return BAG_SUITCASE;

    const int small_object[] = {39, 41, 63, 65, 66, 67, 73};
    if (any_of(raw_class_id, small_object, 7)) return SMALL_OBJECT;

    const int vehicle_bicycle[] = {1, 2, 3};
    if (any_of(raw_class_id, vehicle_bicycle, 3)) return VEHICLE_BICYCLE;

    return GENERIC_OBSTACLE;
}

bool IsObstacleClass(int semantic_class_id)
{
    return semantic_class_id != PERSON;
}

bool IsFurnitureLikeRawClass(int raw_class_id)
{
    if (ModelClassCount() == 25) {
        return IsFurnitureLikeSemantic(SemanticClassFromRaw(raw_class_id));
    }

    const int ids[] = {13, 56, 57, 59, 60};
    return any_of(raw_class_id, ids, 5);
}

bool IsFurnitureLikeSemantic(int semantic_class_id)
{
    return semantic_class_id == CHAIR_SEAT ||
           semantic_class_id == TABLE_DESK ||
           semantic_class_id == SOFA_BED;
}

bool IsSmallObjectSemantic(int semantic_class_id)
{
    return semantic_class_id == SMALL_OBJECT;
}

bool IsVehicleSemantic(int semantic_class_id)
{
    return semantic_class_id == VEHICLE_BICYCLE;
}

float CandidateThreshold(int raw_class_id)
{
    const int sem = SemanticClassFromRaw(raw_class_id);
    if (ModelClassCount() == 25) {
        if (sem == PERSON) return 0.14f;
        if (sem == CHAIR_SEAT) return 0.18f;
        // Detection recall and navigation risk are separate concerns.  Keep
        // vehicle candidates for tracking, then require geometric evidence in
        // the planner before they can trigger an urgent action.
        if (sem == VEHICLE_BICYCLE) return 0.28f;
        if (sem == SMALL_OBJECT) return 0.22f;
        return 0.24f;
    }
    if (sem == PERSON || IsFurnitureLikeSemantic(sem)) return 0.16f;
    if (sem == BAG_SUITCASE || sem == SMALL_OBJECT) return 0.18f;
    if (sem == VEHICLE_BICYCLE) return 0.20f;
    return 0.22f;
}

float RiskWeight(int semantic_class_id)
{
    switch (semantic_class_id) {
        case PERSON: return 1.35f;
        case CHAIR_SEAT: return 1.20f;
        case TABLE_DESK: return 1.15f;
        case SOFA_BED: return 1.20f;
        case BAG_SUITCASE: return 1.05f;
        case SMALL_OBJECT: return 0.80f;
        case VEHICLE_BICYCLE: return 1.30f;
        default: return 1.00f;
    }
}

std::string SemanticLabel(int semantic_class_id)
{
    if (semantic_class_id >= 0 && semantic_class_id < NUM_SEMANTIC_CLASSES) {
        return kSemanticNames[semantic_class_id];
    }
    return kSemanticNames[GENERIC_OBSTACLE];
}

std::string SemanticShortLabel(int semantic_class_id)
{
    if (semantic_class_id >= 0 && semantic_class_id < NUM_SEMANTIC_CLASSES) {
        return kSemanticShortNames[semantic_class_id];
    }
    return kSemanticShortNames[GENERIC_OBSTACLE];
}

std::string RawLabel(int raw_class_id)
{
    if (ModelClassCount() == NUM_SEMANTIC_CLASSES) {
        return SemanticLabel(raw_class_id);
    }
    if (ModelClassCount() == 25) {
        const int n = static_cast<int>(sizeof(kRod25Names) / sizeof(kRod25Names[0]));
        if (raw_class_id >= 0 && raw_class_id < n) {
            return kRod25Names[raw_class_id];
        }
        return "unknown";
    }
    const int n = static_cast<int>(sizeof(kCocoNames) / sizeof(kCocoNames[0]));
    if (raw_class_id >= 0 && raw_class_id < n) {
        return kCocoNames[raw_class_id];
    }
    return "unknown";
}

}  // namespace semantic
}  // namespace obstacle
