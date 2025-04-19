# Makefile для R2R MCP Server

.PHONY: help install install-dev test lint format mypy clean all run

UV = uv

PACKAGE_DIR = app
TESTS_DIR = tests

help:
	@echo "Доступные команды:"
	@echo "make install         - Установить основные зависимости"
	@echo "make install-dev     - Установить зависимости для разработки"
	@echo "make test            - Запустить тесты"
	@echo "make test-cov         - Запустить тесты с отчетом о покрытии"
	@echo "make lint            - Запустить проверку стиля кода (ruff)"
	@echo "make format          - Отформатировать код (black + isort)"
	@echo "make mypy            - Запустить проверку типов"
	@echo "make clean           - Удалить временные файлы"
	@echo "make all             - Запустить lint, format, mypy и test"
	@echo "make run             - Запустить MCP сервер локально"

install:
	$(UV) pip install mcp r2r loguru

install-dev:
	$(UV) pip install ".[dev]"

test:
	$(UV) run pytest $(TESTS_DIR)

test-cov:
	$(UV) run pytest $(TESTS_DIR) --cov=$(PACKAGE_DIR) --cov-report=term-missing

lint:
	$(UV) run ruff check $(PACKAGE_DIR) $(TESTS_DIR)

format:
	$(UV) run black -l 79 $(PACKAGE_DIR) $(TESTS_DIR)
	$(UV) run isort $(PACKAGE_DIR) $(TESTS_DIR)

mypy:
	$(UV) run mypy $(PACKAGE_DIR)

clean:
	rm -rf .pytest_cache
	rm -rf .coverage
	rm -rf __pycache__
	rm -rf $(PACKAGE_DIR)/__pycache__
	rm -rf $(TESTS_DIR)/__pycache__
	rm -rf .mypy_cache
	rm -rf .ruff_cache
	rm -rf dist
	rm -rf build
	rm -rf *.egg-info

all: lint format mypy test

run:
	$(UV) run $(PACKAGE_DIR)/server.py 