import pandas as pd
import pytest
from solution import add_virtual_column
from pipeline import clean_sales_data, compute_revenue_by_region, run_pipeline


class TestAddVirtualColumn:
    def test_chained_operations_and_precedence(self):
        df = pd.DataFrame(
            {"quantity": [10, 5], "unit_price": [20.0, 12.0], "discount": [5.0, 0.0]}
        )
        result = add_virtual_column(
            df, "quantity * unit_price - discount", "revenue"
        )
        expected = pd.DataFrame(
            {
                "quantity": [10, 5],
                "unit_price": [20.0, 12.0],
                "discount": [5.0, 0.0],
                "revenue": [195.0, 60.0],
            }
        )
        assert result.equals(expected)

    def test_parentheses_support(self):
        df = pd.DataFrame({"a": [2, 3], "b": [4, 5], "c": [10, 2]})
        result = add_virtual_column(df, "(a + b) * c", "total")
        assert list(result["total"]) == [60, 16]

    def test_does_not_mutate_input_dataframe(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        result = add_virtual_column(df, "a + b", "c")
        assert list(df.columns) == ["a", "b"]
        assert list(result.columns) == ["a", "b", "c"]

    def test_empty_dataframe_with_valid_columns(self):
        df = pd.DataFrame(columns=["a", "b"])
        result = add_virtual_column(df, "a + b", "c")
        assert list(result.columns) == ["a", "b", "c"]
        assert len(result) == 0

    @pytest.mark.parametrize(
        "invalid_col",
        ["col1", "col-one", "col one", "col$one", "123", ""],
    )
    def test_invalid_existing_column_name_returns_empty(self, invalid_col):
        df = pd.DataFrame([[1, 2]], columns=[invalid_col, "valid_col"])
        result = add_virtual_column(df, "valid_col + valid_col", "new_col")
        assert result.empty

    @pytest.mark.parametrize(
        "invalid_new_col",
        ["new1", "new-col", "new col", "new_col!", ""],
    )
    def test_invalid_new_column_name_returns_empty(self, invalid_new_col):
        df = pd.DataFrame({"a": [1], "b": [2]})
        result = add_virtual_column(df, "a + b", invalid_new_col)
        assert result.empty

    @pytest.mark.parametrize(
        "invalid_role",
        [
            "",
            "   ",
            "a / b",
            "a // b",
            "a % b",
            "a ** b",
            "a + 10",
            "unknown_col + b",
            "a +",
            "* b",
            "a @ b",
        ],
    )
    def test_invalid_role_returns_empty(self, invalid_role):
        df = pd.DataFrame({"a": [10], "b": [2]})
        result = add_virtual_column(df, invalid_role, "c")
        assert result.empty

    def test_non_string_inputs_return_empty(self):
        df = pd.DataFrame({"a": [1], "b": [2]})
        assert add_virtual_column(df, None, "c").empty  # type: ignore[arg-type]
        assert add_virtual_column(df, "a + b", None).empty  # type: ignore[arg-type]


class TestSalesPipeline:
    def test_clean_sales_data_handles_all_anomalies(self):
        raw = pd.DataFrame(
            [
                # Repeated headers
                {
                    "Order ID": "Order ID",
                    "Region": "Region",
                    "Product": "Product",
                    "Quantity ": "Quantity ",
                    "Unit Price": "Unit Price",
                    "Discount": "Discount",
                    "Order Date": "Order Date",
                },
                # Normal row + duplicate
                {
                    "Order ID": "SO-1",
                    "Region": " north  ",
                    "Product": "Mouse",
                    "Quantity ": "10",
                    "Unit Price": "29.50",
                    "Discount": "",
                    "Order Date": "2026-01-01",
                },
                {
                    "Order ID": "SO-1",
                    "Region": " north  ",
                    "Product": "Mouse",
                    "Quantity ": "10",
                    "Unit Price": "29.50",
                    "Discount": "",
                    "Order Date": "2026-01-01",
                },
                # commas decimal price + space in quantity
                {
                    "Order ID": "SO-2",
                    "Region": "EAST",
                    "Product": "Mouse",
                    "Quantity ": "1 000",
                    "Unit Price": "29,50",
                    "Discount": "50.00",
                    "Order Date": "2026-01-02",
                },
                # Missing units price inputed from Mouse (29.50) + negative quantity
                {
                    "Order ID": "SO-3",
                    "Region": "South",
                    "Product": "Mouse",
                    "Quantity ": "-2",
                    "Unit Price": "",
                    "Discount": "",
                    "Order Date": "2026-01-03",
                },
                # TOTAL footer row
                {
                    "Order ID": "TOTAL",
                    "Region": "",
                    "Product": "",
                    "Quantity ": "1008",
                    "Unit Price": "",
                    "Discount": "50.00",
                    "Order Date": "",
                },
            ]
        )

        cleaned = clean_sales_data(raw)
        assert len(cleaned) == 3
        assert list(cleaned["region"]) == ["North", "East", "South"]
        assert list(cleaned["quantity"]) == [10, 1000, -2]
        assert list(cleaned["unit_price"]) == [29.5, 29.5, 29.5]
        assert list(cleaned["discount"]) == [0.0, 50.0, 0.0]

        summary = compute_revenue_by_region(cleaned)
        expected = pd.DataFrame(
            {
                "region": ["East", "North", "South"],
                "revenue": [29450.0, 295.0, -59.0],
            }
        )
        assert summary.equals(expected)

    def test_end_to_end_pipeline_on_sales_csv(self, tmp_path):
        output_file = tmp_path / "revenue_by_region.csv"
        result = run_pipeline("sales.csv", output_file)

        assert output_file.exists()
        assert list(result.columns) == ["region", "revenue"]
        assert list(result["region"]) == ["Central", "East", "North", "South", "West"]
        assert list(result["revenue"]) == [
            515824.67,
            442219.91,
            665897.52,
            504036.04,
            442268.89,
        ]