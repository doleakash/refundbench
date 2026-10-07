import logging

from config.settings import Settings, get_settings


def configure_logging(settings: Settings | None = None) -> None:
	settings = settings or get_settings()
	level = getattr(logging, settings.log_level, None)
	if not isinstance(level, int):
		raise ValueError(f"Invalid LOG_LEVEL: {settings.log_level}")

	logging.basicConfig(
		level=level,
		format="%(asctime)s %(levelname)s %(name)s %(message)s",
	)
