from typing import TypeVar, Optional, Type, get_type_hints

TEntity = TypeVar("TEntity")
TModel = TypeVar("TModel")


class OrmEntityMapper:
    @staticmethod
    def to_entity(model: Optional[TModel], entity_class: Type[TEntity]) -> Optional[TEntity]:
        if model is None:
            return None

        entity_fields = set(get_type_hints(entity_class).keys())

        data = {}
        for k, v in model.__dict__.items():
            if k.startswith("_"):
                continue
            if k in entity_fields:
                data[k] = v

        return entity_class(**data)

    @staticmethod
    def to_model(entity: TEntity, model_cls: Type[TModel]) -> TModel:
        model_fields = set(model_cls.__dict__.keys())

        data = {}
        for k, v in entity.__dict__.items():
            if k.startswith("_"):
                continue
            if k in model_fields:
                data[k] = v

        return model_cls(**data)