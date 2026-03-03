def integrate_functions(func_list):
    results = {}
    for func in func_list:
        try:
            result = func()
            results[func.__name__] = result
        except Exception as e:
            logger.error(f"Function {func.__name__} failed with error: {e}")
            results[func.__name__] = None
    return results