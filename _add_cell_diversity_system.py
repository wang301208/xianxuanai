from pathlib import Path
path = Path('core/cell_diversity.py')
text = path.read_text(encoding='utf-8')
if 'class CellDiversitySystem' not in text:
    addition = "\n\nclass CellDiversitySystem:\n    \"\"\"Facade providing high level access to cell diversity resources.\"\"\"\n\n    def __init__(self):\n        self.database = CellTypeDatabase()\n        self.population_manager = CellPopulationManager(self.database)\n\n    def register_cell_type(self, cell_type: CellType, params: Dict[str, float]) -> None:\n        self.database.register_cell_type(cell_type, params)\n\n    def create_population(self, name: str, cell_type: CellType, size: int) -> Dict[str, Any]:\n        return self.population_manager.create_population(name, cell_type, size)\n\n    def get_population_stats(self, name: str) -> Dict[str, Any]:\n        return self.population_manager.get_population_stats(name)\n\n    def list_available_cell_types(self) -> Dict[str, CellParameters]:\n        return self.database.cell_types\n"
    text = text.rstrip() + addition + "\n"
if '__all__' not in text:
    text = text.rstrip() + "\n__all__ = ['CellType', 'CellParameters', 'CellTypeDatabase', 'CellPopulationManager', 'CellDiversitySystem']\n"
path.write_text(text, encoding='utf-8')
