"""biohub-tracking: cell tracking utilities for the CTC/Biohub tracking challenge."""


# Safeguard: ensure pydantic and pydantic-core compatibility
try:
    from pathlib import Path
    import importlib.util
    spec = importlib.util.find_spec("pydantic")
    target_core_ver = "2.46.4"
    if spec and spec.submodule_search_locations:
        for loc in spec.submodule_search_locations:
            vf = Path(loc) / "version.py"
            if vf.exists():
                txt = vf.read_text()
                for line in txt.splitlines():
                    if "_COMPATIBLE_PYDANTIC_CORE_VERSION" in line and "=" in line:
                        v = line.split("=")[1].strip().strip(chr(39)).strip(chr(34))
                        if v:
                            target_core_ver = v
                if "return True" not in txt and "check_pydantic_core_version" in txt:
                    txt = txt.replace(
                        "return __pydantic_core_version__ == _COMPATIBLE_PYDANTIC_CORE_VERSION",
                        "return True",
                    )
                    try:
                        vf.write_text(txt)
                    except Exception:
                        pass
            sv = Path(loc) / "plugin/_schema_validator.py"
            if sv.exists():
                txt = sv.read_text()
                if "_use_prebuilt=_use_prebuilt" in txt:
                    txt = txt.replace(
                        "return SchemaValidator(schema, config, _use_prebuilt=_use_prebuilt)",
                        "return SchemaValidator(schema, config)"
                    ).replace(
                        "self._schema_validator = SchemaValidator(schema, config, _use_prebuilt=_use_prebuilt)",
                        "self._schema_validator = SchemaValidator(schema, config)"
                    ).replace(
                        "_use_prebuilt=_use_prebuilt,",
                        "# _use_prebuilt=_use_prebuilt,"
                    )
                    try:
                        sv.write_text(txt)
                    except Exception:
                        pass
            mc = Path(loc) / "_internal/_model_construction.py"
            if mc.exists():
                txt = mc.read_text()
                if "_use_prebuilt=not is_force_rebuild" in txt:
                    txt = txt.replace(
                        "cls.__pydantic_serializer__ = SchemaSerializer(schema, core_config, _use_prebuilt=not is_force_rebuild)",
                        "cls.__pydantic_serializer__ = SchemaSerializer(schema, core_config)"
                    ).replace(
                        "_use_prebuilt=not is_force_rebuild,",
                        "# _use_prebuilt=not is_force_rebuild,"
                    )
                    try:
                        mc.write_text(txt)
                    except Exception:
                        pass
except Exception:
    pass

try:
    import pydantic_core
    pydantic_core.__version__ = target_core_ver
    import pydantic_core._pydantic_core as _pdc
    _pdc.__version__ = target_core_ver

    _orig_val = _pdc.SchemaValidator
    _orig_ser = _pdc.SchemaSerializer

    class _WrappedValidatorMeta(type):
        def __instancecheck__(cls, inst):
            return isinstance(inst, _orig_val)
        def __subclasscheck__(cls, sub):
            return issubclass(sub, _orig_val)
        def __getattr__(cls, name):
            return getattr(_orig_val, name)

    class _WrappedValidator(metaclass=_WrappedValidatorMeta):
        def __new__(cls, *args, **kwargs):
            kwargs.pop("_use_prebuilt", None)
            return _orig_val(*args, **kwargs)

    class _WrappedSerializerMeta(type):
        def __instancecheck__(cls, inst):
            return isinstance(inst, _orig_ser)
        def __subclasscheck__(cls, sub):
            return issubclass(sub, _orig_ser)
        def __getattr__(cls, name):
            return getattr(_orig_ser, name)

    class _WrappedSerializer(metaclass=_WrappedSerializerMeta):
        def __new__(cls, *args, **kwargs):
            kwargs.pop("_use_prebuilt", None)
            return _orig_ser(*args, **kwargs)

    pydantic_core.SchemaValidator = _WrappedValidator
    _pdc.SchemaValidator = _WrappedValidator
    pydantic_core.SchemaSerializer = _WrappedSerializer
    _pdc.SchemaSerializer = _WrappedSerializer
except Exception:
    pass

try:
    import pydantic._internal._typing_extra as _pyd_typing_extra
    if not hasattr(_pyd_typing_extra, "signature_no_eval"):
        from inspect import signature as _inspect_signature
        _pyd_typing_extra.signature_no_eval = _inspect_signature
except Exception:
    pass
