#pragma once

#include <string>

namespace obstacle {
namespace semantic {

enum SemanticClass {
    PERSON = 0,
    CHAIR_SEAT = 1,
    TABLE_DESK = 2,
    SOFA_BED = 3,
    BAG_SUITCASE = 4,
    SMALL_OBJECT = 5,
    VEHICLE_BICYCLE = 6,
    GENERIC_OBSTACLE = 7,
    NUM_SEMANTIC_CLASSES = 8
};

int ModelClassCount();
bool IsSupportedRawClass(int raw_class_id);
int SemanticClassFromRaw(int raw_class_id);
bool IsObstacleClass(int semantic_class_id);
bool IsFurnitureLikeRawClass(int raw_class_id);
bool IsFurnitureLikeSemantic(int semantic_class_id);
bool IsSmallObjectSemantic(int semantic_class_id);
bool IsVehicleSemantic(int semantic_class_id);
float CandidateThreshold(int raw_class_id);
float RiskWeight(int semantic_class_id);
std::string SemanticLabel(int semantic_class_id);
std::string SemanticShortLabel(int semantic_class_id);
std::string RawLabel(int raw_class_id);

}  // namespace semantic
}  // namespace obstacle

