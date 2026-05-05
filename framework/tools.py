import inspect
from pydantic import create_model

_TOOL_REGISTRY: dict[str, callable] = {}

def tool_to_schema(func) -> dict:
    sig = inspect.signature(func)
    fields = {
        name: (param.annotation, param.default if param.default != inspect.Parameter.empty else ...)
        for name, param in sig.parameters.items()
    }
    DynamicModel = create_model(f"{func.__name__}_input", **fields)
    schema = DynamicModel.model_json_schema()
    # print("schema: ",schema)
    return {
        "name": func.__name__,
        "description": func.__doc__ or "",
        "input_schema": schema,
    }

def tool(func):
    schema_dict = tool_to_schema(func)
    func.schema = schema_dict
    _TOOL_REGISTRY[func.__name__] = func
    return func

def get_tool(name: str) -> callable:
    if name not in _TOOL_REGISTRY:
        raise KeyError(f"Tool '{name}' not found in registry.")
    return _TOOL_REGISTRY.get(name)

def get_all_schemas() -> list[dict]:
    return [func.schema for func in _TOOL_REGISTRY.values()]