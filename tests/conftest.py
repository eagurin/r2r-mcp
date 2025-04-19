"""
Общие фикстуры и настройки для тестов R2R MCP в каталоге tests

Это локальный файл конфигурации pytest, который автоматически загружается
при запуске тестов и предоставляет настройки и фикстуры только для тестов в этом каталоге.
"""

import warnings

import pytest

# Подавляем предупреждения Pydantic
warnings.filterwarnings(
    "ignore", category=DeprecationWarning, module="pydantic"
)

# Настройка pytest для pytest-asyncio теперь определена в корневом conftest.py
# pytest_plugins = ["pytest_asyncio"]  # Удалено, чтобы избежать ошибки


# Устанавливаем область видимости цикла событий asyncio
def pytest_configure(config):
    """Добавляем настройки командной строки для pytest."""
    # В новых версиях pytest нужно использовать ini_options.markers
    config.addinivalue_line(
        "markers", "asyncio: mark test as an asyncio coroutine"
    )
