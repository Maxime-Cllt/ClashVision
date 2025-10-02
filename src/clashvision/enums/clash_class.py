from enum import Enum


class ClashClass(Enum):
    """Enum representing different Clash of Clans building classes."""

    ELIXIR_STORAGE = 0  # Replace with your actual class name
    GOLD_STORAGE = 1  # Replace with your actual class name

    @staticmethod
    def from_int(value: int) -> "ClashClass":
        """Convert an integer to a ClashClass enum member."""
        for cls in ClashClass:
            if cls.value == value:
                return cls
        raise ValueError(f"No ClashClass with value {value}")

    @property
    def to_color(self) -> str:
        """Map each class to a specific color for visualization."""
        color_map: dict[ClashClass, str] = {
            ClashClass.ELIXIR_STORAGE: "#F461FF",
            ClashClass.GOLD_STORAGE: "#FEE95F",
        }
        return color_map.get(self, "black")  # Default to black if not found

    @staticmethod
    def to_list() -> list[str]:
        """Get all class names as a list of strings."""
        return [cls.name for cls in ClashClass]

    @staticmethod
    def get_palette() -> list[str]:
        """Get all colors in the palette."""
        return [cls.to_color for cls in ClashClass]

    def __str__(self) -> str:
        return self.name.replace("_", " ").title()
