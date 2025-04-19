#!/usr/bin/env python3
"""
Интеграционные тесты для R2R Retrieval System - Model Context Protocol (MCP) Server

Эти тесты проверяют взаимодействие MCP сервера с реальным API R2R.
Для запуска тестов необходимо настроить переменные окружения:
    - R2R_API_URL или R2R_API_KEY

Запуск тестов:
    # Запуск всех тестов (требуется настройка R2R API)
    pytest tests/test_r2r_mcp_integration.py -v

    # Пропуск тестов, требующих реального API
    pytest tests/test_r2r_mcp_integration.py -v -k "not requires_api"
"""

import os
import tempfile
import warnings
from unittest.mock import Mock, patch

import pytest

# Подавляем предупреждения Pydantic о устаревших функциях
warnings.filterwarnings(
    "ignore", category=DeprecationWarning, module="pydantic"
)

# Импортируем тестируемый модуль
from app.server import R2RMCPServer

# Пропускаем тесты, если API не настроен
requires_api = pytest.mark.skipif(
    not (os.environ.get("R2R_API_URL") or os.environ.get("R2R_API_KEY")),
    reason="Требуются настройки R2R API (R2R_API_URL или R2R_API_KEY)",
)


@pytest.fixture
def server():
    """Создаем экземпляр сервера для тестов."""
    # Создаем сервер
    server = R2RMCPServer()

    # Мокаем метод run, чтобы он не запускал реальный сервер
    server.mcp.run = Mock()

    yield server


@pytest.fixture
def test_document():
    """Создает временный тестовый документ для загрузки в R2R."""
    # Создаем временный файл для тестового документа
    with tempfile.NamedTemporaryFile(
        suffix=".txt", delete=False, mode="w+"
    ) as tmp:
        tmp.write(
            """
        # Тестовый документ для R2R
        
        Это тестовый документ, который используется для интеграционных тестов R2R MCP сервера.
        
        Содержит некоторые ключевые слова: Python, FastAPI, R2R, поиск, интеграция, тесты.
        """
        )
        tmp_path = tmp.name

    yield tmp_path

    # Удаляем временный файл после тестов
    try:
        os.unlink(tmp_path)
    except:
        pass


class TestIntegration:
    """Интеграционные тесты R2R MCP Server."""

    @requires_api
    def test_server_initialization(self, server):
        """Тест инициализации сервера с реальными настройками API."""
        # Проверяем, что сервер был создан
        assert server is not None
        assert hasattr(server, "mcp")

        # Проверяем, что клиент может быть получен
        client = server.get_client()
        assert client is not None

    @requires_api
    @pytest.mark.asyncio
    async def test_tools_registration(self, server):
        """Тест проверяет правильную регистрацию всех инструментов."""
        # Проверяем, что все инструменты были зарегистрированы
        tool_names = [tool.name for tool in server.mcp._tools]

        # Проверяем наличие всех ожидаемых инструментов
        expected_tools = [
            "search",
            "rag",
            "web_search",
            "document_search",
            "list_documents",
            "agent_research",
        ]

        for tool_name in expected_tools:
            assert tool_name in tool_names

    @requires_api
    @pytest.mark.asyncio
    async def test_search_integration(self, server):
        """Интеграционный тест инструмента search."""
        # Получаем функцию search из инструментов MCP
        search_tool = None
        for tool in server.mcp._tools:
            if tool.name == "search":
                search_tool = tool.func
                break

        assert search_tool is not None

        # Вызываем инструмент search
        result = await search_tool("Python programming")

        # Проверяем результат
        # Независимо от результатов поиска, ответ должен быть строкой
        assert isinstance(result, str)
        assert len(result) > 0

    @requires_api
    @pytest.mark.asyncio
    async def test_web_search_integration(self, server):
        """Интеграционный тест инструмента web_search."""
        # Получаем функцию web_search из инструментов MCP
        web_search_tool = None
        for tool in server.mcp._tools:
            if tool.name == "web_search":
                web_search_tool = tool.func
                break

        assert web_search_tool is not None

        # Вызываем инструмент web_search
        result = await web_search_tool("latest Python version")

        # Проверяем результат - просто проверяем, что возвращается строка
        # Мы не проверяем содержимое, так как API может иметь ошибки или проблемы с квотой
        assert isinstance(result, str)
        assert len(result) > 0

    @requires_api
    @pytest.mark.asyncio
    async def test_list_documents_integration(self, server):
        """Интеграционный тест инструмента list_documents."""
        # Получаем функцию list_documents из инструментов MCP
        list_documents_tool = None
        for tool in server.mcp._tools:
            if tool.name == "list_documents":
                list_documents_tool = tool.func
                break

        assert list_documents_tool is not None

        # Вызываем инструмент list_documents
        result = await list_documents_tool()

        # Проверяем результат
        assert isinstance(result, str)
        assert "Available Documents" in result

    @requires_api
    @pytest.mark.asyncio
    async def test_document_search_integration(self, server):
        """Интеграционный тест инструмента document_search."""
        # Получаем функцию document_search из инструментов MCP
        document_search_tool = None
        for tool in server.mcp._tools:
            if tool.name == "document_search":
                document_search_tool = tool.func
                break

        assert document_search_tool is not None

        # Вызываем инструмент document_search без document_id
        result = await document_search_tool("Python")

        # Проверяем результат
        assert isinstance(result, str)

    @requires_api
    @pytest.mark.asyncio
    async def test_rag_integration(self, server):
        """Интеграционный тест инструмента rag."""
        # Получаем функцию rag из инструментов MCP
        rag_tool = None
        for tool in server.mcp._tools:
            if tool.name == "rag":
                rag_tool = tool.func
                break

        assert rag_tool is not None

        # Вызываем инструмент rag
        result = await rag_tool("Что такое Python?")

        # Проверяем результат
        assert isinstance(result, str)
        assert len(result) > 0


class TestMCPIntegrationWithMocks:
    """
    Тесты интеграции MCP сервера с моками для определенных компонентов.
    Эти тесты имитируют взаимодействие без необходимости реального API.
    """

    @pytest.mark.asyncio
    async def test_search_with_mocked_client(self):
        """Тест инструмента search с моком клиента."""

        # Создаем мок для клиента R2R
        class MockSearchResults:
            def __init__(self):
                self.chunk_search_results = [
                    type(
                        "MockChunk",
                        (),
                        {
                            "id": "chunk1",
                            "text": "Текст фрагмента 1",
                            "score": 0.95,
                        },
                    )
                ]
                self.web_search_results = []
                self.graph_search_results = []
                self.document_search_results = []

        class MockSearchResponse:
            def __init__(self):
                self.results = MockSearchResults()

        # Создаем мок для метода search
        mock_search = Mock(return_value=MockSearchResponse())

        # Мокаем клиент R2R
        with patch("app.server.R2RClient") as MockR2RClient:
            mock_client = MockR2RClient.return_value
            mock_client.retrieval = Mock()
            mock_client.retrieval.search = mock_search

            # Создаем сервер
            server = R2RMCPServer()
            server.client = mock_client

            # Получаем функцию search из инструментов MCP
            search_tool = None
            for tool in server.mcp._tools:
                if tool.name == "search":
                    search_tool = tool.func
                    break

            assert search_tool is not None

            # Вызываем инструмент search
            result = await search_tool("test query")

            # Проверяем результат
            assert "Vector Search Results" in result
            assert "Текст фрагмента 1" in result
            assert "chunk1" in result

    @pytest.mark.asyncio
    async def test_rag_with_mocked_client(self):
        """Тест инструмента rag с моком клиента."""

        # Создаем мок для результатов RAG
        class MockCitation:
            def __init__(self):
                self.id = "cit1"
                self.payload = {"text": "Текст цитаты"}

        class MockRAGResults:
            def __init__(self):
                self.generated_answer = "Это ответ, сгенерированный RAG"
                self.citations = [MockCitation()]

        class MockRAGResponse:
            def __init__(self):
                self.results = MockRAGResults()

        # Создаем мок для метода rag
        mock_rag = Mock(return_value=MockRAGResponse())

        # Мокаем клиент R2R
        with patch("app.server.R2RClient") as MockR2RClient:
            mock_client = MockR2RClient.return_value
            mock_client.retrieval = Mock()
            mock_client.retrieval.rag = mock_rag

            # Создаем сервер
            server = R2RMCPServer()
            server.client = mock_client

            # Получаем функцию rag из инструментов MCP
            rag_tool = None
            for tool in server.mcp._tools:
                if tool.name == "rag":
                    rag_tool = tool.func
                    break

            # Вызываем инструмент rag
            result = await rag_tool("тестовый вопрос")

            # Проверяем, что метод rag был вызван
            mock_rag.assert_called_once()

            # Проверяем результат
            assert "Это ответ, сгенерированный RAG" in result
            assert "Citations" in result
            assert "[cit1]" in result


if __name__ == "__main__":
    pytest.main(["-v", __file__])
