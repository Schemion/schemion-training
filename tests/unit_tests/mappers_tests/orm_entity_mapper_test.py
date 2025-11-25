from dataclasses import dataclass
from sqlalchemy import Column, Integer, String
from sqlalchemy.orm import declarative_base
from app.infrastructure.mappers import OrmEntityMapper


Base = declarative_base()

@dataclass
class MockEntity:
    id: int
    name: str
    value: int


class MockModel(Base):
    __tablename__ = "mock"

    id = Column(Integer, primary_key=True)
    name = Column(String)
    value = Column(Integer)
    ignored_field = Column(String, default="pls ignore me")

def test_to_entity_maps_matching_fields():
    model = MockModel(id=10, name="hello", value=5)
    model.ignored_field = "skip me"

    entity = OrmEntityMapper.to_entity(model, MockEntity)

    assert isinstance(entity, MockEntity)
    assert entity.id == 10
    assert entity.name == "hello"
    assert entity.value == 5
    assert not hasattr(entity, "ignored_field")

def test_to_model_maps_matching_fields():
    entity = MockEntity(id=123, name="test", value=777)

    model = OrmEntityMapper.to_model(entity, MockModel)

    assert isinstance(model, MockModel)
    assert model.id == 123
    assert model.name == "test"
    assert model.value == 777
    assert model.ignored_field is None or model.ignored_field == "pls ignore me"

def test_to_entity_none_returns_none():
    assert OrmEntityMapper.to_entity(None, MockEntity) is None