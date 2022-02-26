class Storable:
    def __init__(self) -> None:
        pass

    def get_dict_representation(self):
        """
        Get dictionary containing information to re-create class instance
        """
        dict_representation: dict = {}
        return dict_representation
    
    def to_file(self, file_path: str):
        """
        Save class instance to file
        """
        pass

    @classmethod
    def from_file(cls, file_path: str):
        """
        Load class instance from file
        """
        pass