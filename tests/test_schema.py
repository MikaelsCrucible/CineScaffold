from __future__ import annotations

import unittest

from cinescaffold.errors import SchemaValidationError
from cinescaffold.schema import load_schema, validate_model_output
from tests.helpers import ROOT, valid_model_output


class SchemaTest(unittest.TestCase):
    def setUp(self) -> None:
        self.schema = load_schema(ROOT / "schemas/cinematic_brief_model_output.schema.json")

    def test_valid_output_passes(self) -> None:
        validate_model_output(valid_model_output(), self.schema)

    def test_missing_dimension_fails(self) -> None:
        value = valid_model_output()
        del value["camera"]
        with self.assertRaisesRegex(SchemaValidationError, "camera 缺失"):
            validate_model_output(value, self.schema)

    def test_extra_field_fails(self) -> None:
        value = valid_model_output()
        value["unexpected"] = True
        with self.assertRaisesRegex(SchemaValidationError, "额外字段"):
            validate_model_output(value, self.schema)


if __name__ == "__main__":
    unittest.main()
