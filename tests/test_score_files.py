"""Regression tests for the shared model score contract."""
import tempfile
import unittest
from pathlib import Path
from evaluation import load_anomaly_scores, load_evaluation_labels, evaluate_score_files


class ScoreFileTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def write(self, name, text):
        path = self.root / name
        path.write_text(text)
        return path

    def load(self, rows, model="gmm", direction="higher_is_more_suspicious"):
        return load_anomaly_scores(
            self.write("scores.csv", rows), expected_split="validation",
            expected_model=model, score_direction=direction,
        )

    def test_both_models_and_leading_zero_ids(self):
        for model in ("gmm", "ae"):
            result = self.load(
                f"row_id,split,model,anomaly_score\n002,validation,{model},-1\n001,validation,{model},2\n",
                model=model,
            )
            self.assertEqual(result.row_id.tolist(), ["001", "002"])
            self.assertEqual(result.anomaly_score.tolist(), [2, -1])

    def test_invalid_score_files(self):
        header = "row_id,split,model,anomaly_score\n"
        for rows in ("a,validation,gmm,1\na,validation,gmm,2", ",validation,gmm,1",
                     "a,test,gmm,1", "a,validation,ae,1", "a,validation,gmm,inf",
                     "a,validation,gmm,NaN", "a,validation,gmm,hello",
                     " a,validation,gmm,1", "a,validation,gmm,1,extra", ""):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                self.load(header + rows)
        for columns in ("row_id,split,model,anomaly_score,Class",
                        "row_id,split,model,anomaly_score,row_id"):
            with self.assertRaises(ValueError):
                self.load(columns + "\na,validation,gmm,1,0\n")

    def test_explicit_direction(self):
        with self.assertRaises(ValueError):
            self.load("row_id,split,model,anomaly_score\na,validation,gmm,1\n",
                      direction="lower_is_more_suspicious")

    def files(self):
        paths = []
        for split in ("validation", "test"):
            prefix = split[0]
            paths.extend([
                self.write(split + "_scores.csv", "row_id,split,model,anomaly_score\n"
                           f"{prefix}2,{split},gmm,1\n{prefix}1,{split},gmm,1\n"),
                self.write(split + "_labels.csv", "row_id,split,Class\n"
                           f"{prefix}2,{split},0\n{prefix}1,{split},1\n"),
            ])
        return paths

    def evaluate(self, paths):
        return evaluate_score_files(
            *paths, model="gmm", score_direction="higher_is_more_suspicious",
            fpr_budgets=(0.0,), review_capacities=(1,),
        )

    def test_alignment_and_ties_ignore_file_order(self):
        paths = self.files()
        result = self.evaluate(paths)
        self.assertEqual(result.test_triage.iloc[0]["frauds_caught"], 1)
        self.assertEqual(result.ranking_metrics.iloc[0]["average_precision_ap"], 0.5)
        self.assertEqual(result.selections[0].validation_recall, 0)
        paths[2].write_text("row_id,split,model,anomaly_score\nt1,test,gmm,1\nt2,test,gmm,1\n")
        self.assertTrue(result.test_triage.equals(self.evaluate(paths).test_triage))

    def test_missing_or_extra_label_ids(self):
        for rows in ("v1,validation,1", "v1,validation,1\nv3,validation,0",
                     "v1,validation,1\nv2,validation,0\nv3,validation,0"):
            paths = self.files()
            paths[1].write_text("row_id,split,Class\n" + rows + "\n")
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                self.evaluate(paths)

    def test_invalid_labels(self):
        for rows in ("a,validation,2", "a,test,1", "a,validation,1\na,validation,0",
                     ",validation,1"):
            path = self.write("labels.csv", "row_id,split,Class\n" + rows + "\n")
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                load_evaluation_labels(path, expected_split="validation")
        with self.assertRaises(ValueError):
            load_evaluation_labels(path, expected_split="train")

    def test_ae_uses_the_same_evaluation_adapter(self):
        paths = self.files()
        for path in (paths[0], paths[2]):
            path.write_text(path.read_text().replace(",gmm,", ",ae,"))
        result = evaluate_score_files(
            *paths, model="ae", score_direction="higher_is_more_suspicious",
            fpr_budgets=(0.0,), review_capacities=(1,),
        )
        self.assertEqual(result.test_triage.iloc[0]["frauds_caught"], 1)

    def test_label_order_does_not_change_metrics(self):
        paths = self.files()
        original = self.evaluate(paths)
        paths[3].write_text("row_id,split,Class\nt1,test,1\nt2,test,0\n")
        reordered = self.evaluate(paths)
        self.assertTrue(original.ranking_metrics.equals(reordered.ranking_metrics))
        self.assertTrue(original.test_triage.equals(reordered.test_triage))

    def test_overlapping_split_ids(self):
        paths = self.files()
        paths[2].write_text("row_id,split,model,anomaly_score\nv1,test,gmm,2\nv2,test,gmm,1\n")
        paths[3].write_text("row_id,split,Class\nv1,test,1\nv2,test,0\n")
        with self.assertRaises(ValueError):
            self.evaluate(paths)


if __name__ == "__main__":
    unittest.main()
