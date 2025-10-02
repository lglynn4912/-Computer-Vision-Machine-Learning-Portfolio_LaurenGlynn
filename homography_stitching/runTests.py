def run_tests(test_name, fun_handles):
    execution_list = filter_tests(test_name, fun_handles)
    for exec_fcn in execution_list:
        print(f'Executing {exec_fcn.__name__}')
        exec_fcn()

def filter_tests(test_name, fun_handles):
    if test_name == 'all':
        return [fun_handles[name] for name in fun_handles 
                          if not name.startswith('demo')]
    else:
        if test_name in fun_handles:
            return [fun_handles[test_name]]
    
    print('===============')
    print('Registered tests are:')
    for test in fun_handles.keys():
        print(f"- {test}")

    return []