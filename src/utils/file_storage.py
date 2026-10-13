import uuid
from pathlib import Path
from src.config import config
from src.utils.logger import logger

class FileStorage:
    @staticmethod
    def get_temp_filepath(filename: str, user_id: int | str) -> Path:
        safe_filename = Path(filename).name
        unique_prefix = f"{user_id}_{uuid.uuid4().hex[:8]}"
        destination = config.TEMP_DIR / f"{unique_prefix}_{safe_filename}"
        return destination

    @staticmethod
    def cleanup_file(filepath: Path | str) -> None:
        try:
            path = Path(filepath)
            if path.exists():
                path.unlink()
                logger.debug(f"Temporary file removed: {path}")
        except Exception as e:
            logger.warning(f"Failed to remove temporary file {filepath}: {e}")
