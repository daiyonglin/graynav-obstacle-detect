import unittest

from training.scripts.download_coco_indoor8_subset import select_subset


class CocoIndoor8SubsetTest(unittest.TestCase):
    def test_selection_is_balanced_deterministic_and_bounded(self) -> None:
        categories = [
            {"id": index + 1, "name": name}
            for index, name in enumerate((
                "person", "chair", "dining table", "backpack",
                "handbag", "suitcase", "couch", "bench",
            ))
        ]
        images = [{"id": index, "file_name": f"{index}.jpg"} for index in range(1, 81)]
        annotations = []
        annotation_id = 1
        for image in images[:72]:
            class_index = (int(image["id"]) - 1) % 8
            annotations.append({
                "id": annotation_id,
                "image_id": image["id"],
                "category_id": class_index + 1,
                "iscrowd": 0,
            })
            annotation_id += 1
        payload = {"categories": categories, "images": images, "annotations": annotations}
        quotas = {row["name"]: 3 for row in categories}
        first = select_subset(payload, quotas, max_images=40, negative_images=4, seed=42)
        second = select_subset(payload, quotas, max_images=40, negative_images=4, seed=42)
        self.assertEqual(first, second)
        selected, counts, negatives = first
        self.assertLessEqual(len(selected), 40)
        self.assertEqual(len(negatives), 4)
        for name in quotas:
            self.assertGreaterEqual(counts[name], 3)


if __name__ == "__main__":
    unittest.main()
