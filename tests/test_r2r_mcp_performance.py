#!/usr/bin/env python3
"""
Тесты производительности для R2R Retrieval System - Model Context Protocol (MCP) Server

Эти тесты измеряют время выполнения различных операций MCP сервера
и позволяют оценить эффективность работы компонентов системы.

Запуск тестов:
    # Запуск всех тестов производительности
    pytest tests/test_r2r_mcp_performance.py -v

    # Запуск определенной категории тестов
    pytest tests/test_r2r_mcp_performance.py::TestFormatting -v
"""

import statistics
import time
import warnings
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

# Подавляем предупреждения Pydantic о устаревших функциях
warnings.filterwarnings(
    "ignore", category=DeprecationWarning, module="pydantic"
)

# Импортируем тестируемый модуль
from app.server import (
    R2RMCPServer,
    format_rag_response,
    format_search_results_for_llm,
    id_to_shorthand,
)


# Только для внутреннего использования в тестах
def measure_execution_time(func, *args, **kwargs):
    """Измеряет время выполнения функции."""
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    return result, end_time - start_time


# Только для внутреннего использования в тестах
async def measure_async_execution_time(func, *args, **kwargs):
    """Измеряет время выполнения асинхронной функции."""
    start_time = time.time()
    result = await func(*args, **kwargs)
    end_time = time.time()
    return result, end_time - start_time


class TestFormatting:
    """Тесты производительности форматирования результатов."""

    def test_id_to_shorthand_performance(self):
        """Тест производительности функции id_to_shorthand."""
        # Генерируем большое количество ID разной длины
        ids = [str(i) * (i % 50 + 1) for i in range(1, 1001)]

        # Измеряем время выполнения
        times = []
        for id in ids:
            _, execution_time = measure_execution_time(id_to_shorthand, id)
            times.append(execution_time)

        # Анализируем результаты
        avg_time = statistics.mean(times)
        max_time = max(times)
        min_time = min(times)

        print("\nid_to_shorthand performance:")
        print(f"  Average time: {avg_time:.6f}s")
        print(f"  Min time: {min_time:.6f}s")
        print(f"  Max time: {max_time:.6f}s")

        # Проверяем, что среднее время выполнения меньше порогового значения
        assert avg_time < 0.001  # Должно быть очень быстро

    def test_format_search_results_performance_empty(self):
        """Тест производительности форматирования пустых результатов поиска."""

        # Создаем мок-объект с пустыми результатами поиска
        class MockEmptyResults:
            def __init__(self):
                self.chunk_search_results = []
                self.graph_search_results = []
                self.web_search_results = []
                self.document_search_results = []

        results = MockEmptyResults()

        # Измеряем время выполнения 100 раз
        times = []
        for _ in range(100):
            _, execution_time = measure_execution_time(
                format_search_results_for_llm, results
            )
            times.append(execution_time)

        # Анализируем результаты
        avg_time = statistics.mean(times)
        max_time = max(times)
        min_time = min(times)

        print("\nformat_search_results_for_llm (empty) performance:")
        print(f"  Average time: {avg_time:.6f}s")
        print(f"  Min time: {min_time:.6f}s")
        print(f"  Max time: {max_time:.6f}s")

        # Проверяем, что среднее время выполнения меньше порогового значения
        assert avg_time < 0.005  # Должно быть быстро для пустых результатов

    def test_format_search_results_performance_with_data(self):
        """Тест производительности форматирования результатов поиска с данными."""

        # Создаем мок-объекты с результатами поиска
        class MockChunk:
            def __init__(self, id, text, score):
                self.id = id
                self.text = text
                self.score = score

        class MockGraph:
            def __init__(self, id):
                self.id = id
                self.content = type(
                    "Content",
                    (),
                    {
                        "name": f"Entity {id}",
                        "description": f"Description for entity {id}",
                    },
                )

        class MockWeb:
            def __init__(self, id):
                self.id = id
                self.title = f"Web result {id}"
                self.link = f"https://example.com/{id}"
                self.snippet = f"This is a snippet for web result {id}" * 5

        class MockDoc:
            def __init__(self, id):
                self.id = id
                self.title = f"Document {id}"
                self.summary = f"Summary for document {id}" * 10
                self.chunks = [
                    {"id": f"chunk_{id}_{i}", "text": f"Chunk text {i}" * 50}
                    for i in range(1, 4)
                ]

        class MockResults:
            def __init__(self, num_results=5):
                self.chunk_search_results = [
                    MockChunk(
                        f"chunk{i}", f"Chunk text {i}" * 100, 0.95 - (i * 0.05)
                    )
                    for i in range(1, num_results + 1)
                ]
                self.graph_search_results = [
                    MockGraph(f"graph{i}") for i in range(1, num_results + 1)
                ]
                self.web_search_results = [
                    MockWeb(f"web{i}") for i in range(1, num_results + 1)
                ]
                self.document_search_results = [
                    MockDoc(f"doc{i}")
                    for i in range(
                        1, 4
                    )  # Меньше документов, так как они содержат больше данных
                ]

        # Создаем результаты с разным количеством данных
        small_results = MockResults(num_results=2)
        medium_results = MockResults(num_results=5)
        large_results = MockResults(num_results=10)

        # Измеряем время выполнения для каждого размера данных
        for name, results in [
            ("small", small_results),
            ("medium", medium_results),
            ("large", large_results),
        ]:
            times = []
            for _ in range(20):  # 20 итераций для каждого размера
                _, execution_time = measure_execution_time(
                    format_search_results_for_llm, results
                )
                times.append(execution_time)

            # Анализируем результаты
            avg_time = statistics.mean(times)
            max_time = max(times)
            min_time = min(times)

            print(f"\nformat_search_results_for_llm ({name}) performance:")
            print(f"  Average time: {avg_time:.6f}s")
            print(f"  Min time: {min_time:.6f}s")
            print(f"  Max time: {max_time:.6f}s")

            # Проверяем, что среднее время выполнения меньше порогового значения
            # Пороговые значения зависят от размера данных
            if name == "small":
                assert avg_time < 0.01
            elif name == "medium":
                assert avg_time < 0.02
            else:  # large
                assert avg_time < 0.05

    def test_format_rag_response_performance(self):
        """Тест производительности форматирования ответа RAG."""

        # Создаем мок-объекты с результатами RAG
        class MockCitation:
            def __init__(self, id, text):
                self.id = id
                self.payload = text

        class MockRAGResults:
            def __init__(self, num_citations=5):
                self.generated_answer = "This is a generated answer" * 50
                self.citations = [
                    MockCitation(f"cit{i}", f"Citation text {i}" * 20)
                    for i in range(1, num_citations + 1)
                ]

        class MockRAGResponse:
            def __init__(self, num_citations=5):
                self.results = MockRAGResults(num_citations)

        # Создаем ответы с разным количеством цитат
        small_response = MockRAGResponse(num_citations=2)
        medium_response = MockRAGResponse(num_citations=5)
        large_response = MockRAGResponse(num_citations=10)

        # Измеряем время выполнения для каждого размера данных
        for name, response in [
            ("small", small_response),
            ("medium", medium_response),
            ("large", large_response),
        ]:
            times = []
            for _ in range(50):  # 50 итераций для каждого размера
                _, execution_time = measure_execution_time(
                    format_rag_response, response
                )
                times.append(execution_time)

            # Анализируем результаты
            avg_time = statistics.mean(times)
            max_time = max(times)
            min_time = min(times)

            print(f"\nformat_rag_response ({name}) performance:")
            print(f"  Average time: {avg_time:.6f}s")
            print(f"  Min time: {min_time:.6f}s")
            print(f"  Max time: {max_time:.6f}s")

            # Проверяем, что среднее время выполнения меньше порогового значения
            assert (
                avg_time < 0.02
            )  # Должно быть быстро даже для больших ответов


@pytest.fixture
def mock_r2r_client():
    """Фикстура для создания мока клиента R2R."""
    # Создаем многоуровневый мок для клиента R2R
    mock_client = MagicMock()

    # Настраиваем retrieval объект
    mock_client.retrieval = MagicMock()
    mock_client.retrieval.search = MagicMock()
    mock_client.retrieval.rag = MagicMock()
    mock_client.retrieval.web_search = MagicMock()
    mock_client.retrieval.agent = MagicMock()

    # Настраиваем documents объект
    mock_client.documents = MagicMock()
    mock_client.documents.list = MagicMock()

    return mock_client


@pytest.fixture
def mcp_server(mock_r2r_client):
    """Фикстура для создания экземпляра сервера MCP с моком клиента."""
    with patch("app.server.R2RClient", return_value=mock_r2r_client):
        with patch("app.server.R2RAsyncClient", return_value=AsyncMock()):
            server = R2RMCPServer()

            # Встраиваем мок клиента напрямую
            server.client = mock_r2r_client

            # Мокаем метод run, чтобы он не запускал реальный сервер
            server.mcp.run = Mock()

            yield server


class TestToolPerformance:
    """Тесты производительности инструментов MCP."""

    @pytest.mark.asyncio
    async def test_search_performance(self, mcp_server, mock_r2r_client):
        """Тест производительности инструмента search."""

        # Настраиваем мок для возврата результатов поиска
        class MockResults:
            def __init__(self):
                self.chunk_search_results = []
                self.graph_search_results = []
                self.web_search_results = []
                self.document_search_results = []

        class MockResponse:
            def __init__(self):
                self.results = MockResults()

        mock_r2r_client.retrieval.search.return_value = MockResponse()

        # Получаем функцию search из инструментов MCP
        search_tool = None
        for tool in mcp_server.tools:
            if tool.name == "search":
                search_tool = tool
                break

        assert search_tool is not None

        # Измеряем производительность
        start_time = time.time()
        for _ in range(100):
            result = await search_tool("test query")
            assert isinstance(result, str)
        end_time = time.time()

        # Проверяем, что среднее время выполнения не превышает 50мс
        avg_time = (end_time - start_time) / 100
        assert (
            avg_time < 0.05
        ), f"Среднее время выполнения search: {avg_time:.3f}с"

    @pytest.mark.asyncio
    async def test_web_search_performance(self, mcp_server, mock_r2r_client):
        """Тест производительности инструмента web_search."""

        # Настраиваем мок для возврата результатов веб-поиска
        class MockWebResult:
            def __init__(self, title, link, snippet):
                self.title = title
                self.link = link
                self.snippet = snippet

        class MockResults:
            def __init__(self):
                self.web_search_results = [
                    MockWebResult(
                        f"Title {i}",
                        f"https://example.com/{i}",
                        f"Snippet {i}",
                    )
                    for i in range(1, 6)
                ]

        class MockResponse:
            def __init__(self):
                self.results = MockResults()

        mock_r2r_client.retrieval.web_search.return_value = MockResponse()

        # Получаем функцию web_search из инструментов MCP
        web_search_tool = None
        for tool in mcp_server.tools:
            if tool.name == "web_search":
                web_search_tool = tool
                break

        assert web_search_tool is not None

        # Измеряем производительность
        start_time = time.time()
        for _ in range(100):
            result = await web_search_tool("test query")
            assert isinstance(result, str)
        end_time = time.time()

        # Проверяем, что среднее время выполнения не превышает 50мс
        avg_time = (end_time - start_time) / 100
        assert (
            avg_time < 0.05
        ), f"Среднее время выполнения web_search: {avg_time:.3f}с"


if __name__ == "__main__":
    pytest.main(["-v", __file__])
