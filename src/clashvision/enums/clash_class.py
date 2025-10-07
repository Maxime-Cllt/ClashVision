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
    def to_hex(self) -> str:
        """Map each class to a specific color for visualization."""
        color_map: dict[ClashClass, str] = {
            ClashClass.ELIXIR_STORAGE: "#F461FF",
            ClashClass.GOLD_STORAGE: "#FFD700",
        }
        return color_map.get(self, "#000000")  # Default to black if not found

    @property
    def to_rgb(self) -> tuple[int, ...]:
        """Get the RGB tuple for the class color."""
        hex_color = self.to_hex.lstrip("#")
        return tuple(int(hex_color[i: i + 2], 16) for i in (0, 2, 4))

    @property
    def to_bgr(self) -> tuple[int, ...]:
        """Get the BGR tuple for OpenCV (reversed RGB)."""
        rgb = self.to_rgb
        return (rgb[2], rgb[1], rgb[0])  # Reverse RGB to BGR

    @staticmethod
    def to_list() -> list[str]:
        """Get all class names as a list of strings."""
        return [cls.name for cls in ClashClass]

    @staticmethod
    def get_palette() -> list[str]:
        """Get all colors in the palette."""
        return [cls.to_hex for cls in ClashClass]  # Fixed: was missing parentheses

    def __str__(self) -> str:
        return self.name.replace("_", " ").title()
