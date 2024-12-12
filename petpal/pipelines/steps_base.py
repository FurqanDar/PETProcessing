import inspect
from typing import Callable


class ArgsDict(dict):
    """
    A specialized subclass of Python's built-in `dict` that provides a customized string representation.

    Attributes:
        None, since ArgsDict inherits directly from dict.
    """
    def __str__(self):
        """
        Returns a formatted string representation of the dictionary.
        
        The string output will list each key-value pair on a new line with indentation to improve readability.
        
        Returns:
            str: A string containing the formatted key-value pairs, indented for clarity.
        """
        rep_str = [f'    {arg}={val}' for arg, val in self.items()]
        return ',\n'.join(rep_str)


class StepsAPI:
    """
    StepsAPI provides an interface for defining steps in a processing pipeline.

    This class outlines methods that allow input and output management between different steps,
    and perform inference of output files based on input data and given parameters.

    """
    
    def set_input_as_output_from(self, sending_step):
        """
        Sets the input of the current step as the output from a specified sending step.

        Args:
            sending_step: The previous step from which the output will be used as input for the current step.

        Raises:
            NotImplementedError: This method should be implemented by subclasses.
            
        Notes:
            For a concrete example, take a look at:
            :meth:`TACsFromSegmentationStep<petpal.pipelines.preproc_steps.TACsFromSegmentationStep.set_input_as_output_from>`
        """
        raise NotImplementedError
    
    def infer_outputs_from_inputs(self, out_dir: str, der_type: str, suffix: str = None, ext: str = None, **extra_desc):
        """
        Infers output files from input data based on the specified output directory,
        derivative type, optional suffix and extension, plus any extra descriptions.

        Args:
            out_dir (str): The directory where the output files will be saved.
            der_type (str): The type of derivative being produced.
            suffix (str, optional): An optional suffix for the output files.
            ext (str, optional): An optional extension for the output files.
            **extra_desc: Additional keyword arguments for extra descriptions to be included.

        Raises:
            NotImplementedError: This method should be implemented by subclasses.
            
        Notes:
            For a concrete example, take a look at:
            :meth:`TACsFromSegmentationStep<petpal.pipelines.preproc_steps.TACsFromSegmentationStep.infer_outputs_from_inputs>`
        """
        raise NotImplementedError


class FunctionBasedStep(StepsAPI):
    """
    A step in a processing pipeline based on a callable function.

    This class allows for the execution of a given function with specified arguments and keyword arguments,
    validating that all mandatory parameters are provided.

    Attributes:
        name (str): The name of the step.
        function (Callable): The function to be executed in this step.
        args (tuple): Positional arguments to be passed to the function.
        kwargs (ArgsDict): Keyword arguments to be passed to the function.
        func_sig (inspect.Signature): The signature of the function for validating arguments.

    """
    def __init__(self, name: str, function: Callable, *args, **kwargs) -> None:
        """
        Initializes a function-based step in the processing pipeline.

        Args:
            name (str): The name of the step.
            function (Callable): The function to be executed in this step.
            *args: Positional arguments to be passed to the function.
            **kwargs: Keyword arguments to be passed to the function.
            
        """
        self.name = name
        self.function = function
        self._func_name = function.__name__
        self.args = args
        self.kwargs = ArgsDict(kwargs)
        self.func_sig = inspect.signature(self.function)
        self.validate_kwargs_for_non_default_have_been_set()
    
    def get_function_args_not_set_in_kwargs(self) -> ArgsDict:
        """
        Retrieves arguments of the function that are not set in the keyword arguments.

        Returns:
            ArgsDict: A dictionary of function arguments that have not been set in the keyword arguments.
        """
        unset_args_dict = ArgsDict()
        func_params = self.func_sig.parameters
        arg_names = list(func_params)
        for arg_name in arg_names[len(self.args):]:
            if arg_name not in self.kwargs and arg_name != 'kwargs':
                unset_args_dict[arg_name] = func_params[arg_name].default
        return unset_args_dict
    
    def get_empty_default_kwargs(self) -> list:
        """
        Identifies arguments that have not been provided and lack default values.

        Returns:
            list: A list of argument names that have no default values and are not provided.
        """
        unset_args_dict = self.get_function_args_not_set_in_kwargs()
        empty_kwargs = []
        for arg_name, arg_val in unset_args_dict.items():
            if arg_val is inspect.Parameter.empty:
                if arg_name not in self.kwargs:
                    empty_kwargs.append(arg_name)
        return empty_kwargs
    
    def validate_kwargs_for_non_default_have_been_set(self) -> None:
        """
        Validates that all mandatory arguments have been provided.

        Raises:
            RuntimeError: If any mandatory arguments are missing.
        """
        empty_kwargs = self.get_empty_default_kwargs()
        if empty_kwargs:
            unset_args = '\n'.join(empty_kwargs)
            raise RuntimeError(f"For {self._func_name}, the following arguments must be set:\n{unset_args}")
    
    def execute(self):
        """
        Executes the function with the provided arguments and keyword arguments.

        Raises:
            The function may raise any exceptions that its implementation can throw.
        """
        print(f"(Info): Executing {self.name}")
        self.function(*self.args, **self.kwargs)
        print(f"(Info): Finished {self.name}")
    
    def generate_kwargs_from_args(self) -> ArgsDict:
        """
        Converts positional arguments into keyword arguments.

        Returns:
            ArgsDict: A dictionary where positional arguments are mapped to their corresponding parameter names.
        """
        args_to_kwargs_dict = ArgsDict()
        for arg_name, arg_val in zip(list(self.func_sig.parameters), self.args):
            args_to_kwargs_dict[arg_name] = arg_val
        return args_to_kwargs_dict
    
    def __str__(self):
        """
        Returns a detailed string representation of the FunctionBasedStep instance.

        Returns:
            str: A string describing the step, including its name, function, arguments, and keyword arguments.
        """
        args_to_kwargs_dict = self.generate_kwargs_from_args()
        info_str = [f'({type(self).__name__} Info):',
                    f'Step Name: {self.name}',
                    f'Function Name: {self._func_name}',
                    f'Arguments Passed:',
                    f'{args_to_kwargs_dict if args_to_kwargs_dict else "N/A"}',
                    'Keyword-Arguments Set:',
                    f'{self.kwargs if self.kwargs else "N/A"}',
                    'Default Arguments:',
                    f'{self.get_function_args_not_set_in_kwargs()}']
        return '\n'.join(info_str)
    
    def __repr__(self):
        """
        Returns an unambiguous string representation of the FunctionBasedStep instance.

        Returns:
            str: A string representation showing how the FunctionBasedStep can be recreated.
        """
        cls_name = type(self).__name__
        full_func_name = f'{self.function.__module__}.{self._func_name}'
        info_str = [f'{cls_name}(', f'name={repr(self.name)},', f'function={full_func_name},']
        
        init_params = inspect.signature(self.__init__).parameters
        for arg_name in list(init_params)[2:-2]:
            info_str.append(f'{arg_name}={repr(getattr(self, arg_name))},')
        
        for arg_name, arg_val in zip(list(self.func_sig.parameters), self.args):
            info_str.append(f'{arg_name}={repr(arg_val)}', )
        
        for arg_name, arg_val in self.kwargs.items():
            info_str.append(f'{arg_name}={repr(arg_val)},')
        
        info_str.append(')')
        
        return f'\n    '.join(info_str)
    
    def all_args_non_empty_strings(self):
        """
        Checks if all positional arguments are non-empty strings.

        Returns:
            bool: True if all positional arguments are non-empty strings, False otherwise.
        """
        for arg in self.args:
            if arg == '':
                return False
        return True
    
    def all_kwargs_non_empty_strings(self):
        """
        Checks if all keyword arguments are non-empty strings.

        Returns:
            bool: True if all keyword arguments are non-empty strings, False otherwise.
        """
        for arg_name, arg in self.kwargs.items():
            if arg == '':
                return False
        return True
    
    def can_potentially_run(self):
        """
        Determines if the step can potentially be executed based on argument validation.
        Very simply checks if all arguments and keyword arguments are non-empty strings.

        Returns:
            bool: True if the step can potentially run, False otherwise.
        """
        return self.all_args_non_empty_strings() and self.all_kwargs_non_empty_strings()


class ObjectBasedStep(StepsAPI):
    """
    A step in a processing pipeline that is based on instantiating and invoking methods on an object.

    This class allows for initialization and execution of a specified object with given arguments and keyword arguments,
    validating that all mandatory parameters are provided.

    Attributes:
        name (str): The name of the step.
        class_type (type): The class type to be instantiated in this step.
        init_kwargs (ArgsDict): Keyword arguments for initializing the class.
        call_kwargs (ArgsDict): Keyword arguments for invoking the class.
        init_sig (inspect.Signature): The initialization signature of the class for validating arguments.
        call_sig (inspect.Signature): The call signature of the class for validating arguments.

    """
    def __init__(self, name: str, class_type: type, init_kwargs: dict, call_kwargs: dict) -> None:
        """
        Initializes an object-based step in the processing pipeline.

        Args:
            name (str): The name of the step.
            class_type (type): The class type to be instantiated in this step.
            init_kwargs (dict): Keyword arguments for initializing the class.
            call_kwargs (dict): Keyword arguments for invoking the class.
        """
        self.name: str = name
        self.class_type: type = class_type
        self.init_kwargs: ArgsDict = ArgsDict(init_kwargs)
        self.call_kwargs: ArgsDict = ArgsDict(call_kwargs)
        self.init_sig: inspect.Signature = inspect.signature(self.class_type.__init__)
        self.call_sig: inspect.Signature = inspect.signature(self.class_type.__call__)
        self.validate_kwargs()
    
    def validate_kwargs(self):
        """
        Validates that all mandatory initialization and call arguments have been provided.

        Raises:
            RuntimeError: If any mandatory arguments are missing.
        """
        empty_init_kwargs = self.get_empty_default_kwargs(self.init_sig, self.init_kwargs)
        empty_call_kwargs = self.get_empty_default_kwargs(self.call_sig, self.call_kwargs)
        
        if empty_init_kwargs or empty_call_kwargs:
            err_msg = [f"For {self.class_type.__name__}, the following arguments must be set:"]
            if empty_init_kwargs:
                err_msg.append("Initialization:")
                err_msg.append(f"{empty_init_kwargs}")
            if empty_call_kwargs:
                err_msg.append("Calling:")
                err_msg.append(f"{empty_call_kwargs}")
            raise RuntimeError("\n".join(err_msg))
    
    @staticmethod
    def get_args_not_set_in_kwargs(sig: inspect.Signature, kwargs: dict) -> dict:
        """
        Retrieves arguments of the signature that are not set in the keyword arguments.

        Args:
            sig (inspect.Signature): The signature of the function or method.
            kwargs (dict): The keyword arguments provided.

        Returns:
            dict: A dictionary of arguments that are not set in the keyword arguments.
        """
        unset_args_dict = ArgsDict()
        for arg_name, arg_val in sig.parameters.items():
            if arg_name not in kwargs and arg_name != 'self':
                unset_args_dict[arg_name] = arg_val.default
        return unset_args_dict
    
    def get_empty_default_kwargs(self, sig: inspect.Signature, set_kwargs: dict) -> list:
        """
        Identifies arguments that have not been provided and lack default values.

        Args:
            sig (inspect.Signature): The signature of the function or method.
            set_kwargs (dict): The keyword arguments provided.

        Returns:
            list: A list of argument names that have no default values and are not provided.
        """
        unset_kwargs = self.get_args_not_set_in_kwargs(sig=sig, kwargs=set_kwargs)
        empty_kwargs = []
        for arg_name, arg_val in unset_kwargs.items():
            if arg_val is inspect.Parameter.empty:
                if arg_name not in set_kwargs:
                    empty_kwargs.append(arg_name)
        return empty_kwargs
    
    def execute(self) -> None:
        """
        Instantiates the class and invokes it with the provided arguments.

        Raises:
            The function may raise any exceptions that its implementation can throw.
        """
        print(f"(Info): Executing {self.name}")
        obj_instance = self.class_type(**self.init_kwargs)
        obj_instance(**self.call_kwargs)
        print(f"(Info): Finished {self.name}")
    
    def __str__(self):
        """
        Returns a detailed string representation of the ObjectBasedStep instance.

        Returns:
            str: A string describing the step, including its name, class, initialization, and call arguments.
        """
        unset_init_args = self.get_args_not_set_in_kwargs(self.init_sig, self.init_kwargs)
        unset_call_args = self.get_args_not_set_in_kwargs(self.call_sig, self.call_kwargs)
        
        info_str = [f'({type(self).__name__} Info):', f'Step Name: {self.name}',
                    f'Class Name: {self.class_type.__name__}', 'Initialization Arguments:', f'{self.init_kwargs}',
                    'Default Initialization Arguments:', f'{unset_init_args if unset_init_args else "N/A"}',
                    'Call Arguments:', f'{self.call_kwargs if self.call_kwargs else "N/A"}', 'Default Call Arguments:',
                    f'{unset_call_args if unset_call_args else "N/A"}']
        return '\n'.join(info_str)
    
    def __repr__(self):
        """
        Returns an unambiguous string representation of the ObjectBasedStep instance.

        Returns:
            str: A string representation showing how the ObjectBasedStep can be recreated.
        """
        cls_name = type(self).__name__
        full_func_name = f'{self.class_type.__module__}.{self.class_type.__name__}'
        info_str = [f'{cls_name}(', f'name={repr(self.name)},', f'class_type={full_func_name},']
        
        if self.init_kwargs:
            info_str.append('init_kwargs={')
            for arg_name, arg_val in self.init_kwargs.items():
                info_str.append(f'    {arg_name}={repr(arg_val)},')
            info_str[-1] = f'{info_str[-1]}' + '}'
        
        if self.call_kwargs:
            info_str.append('call_kwargs={')
            for arg_name, arg_val in self.call_kwargs.items():
                info_str.append(f'    {arg_name}={repr(arg_val)},')
            info_str[-1] = f'{info_str[-1]}' + '}'
        
        info_str.append(')')
        
        return f'\n    '.join(info_str)
    
    def all_init_kwargs_non_empty_strings(self):
        """
        Checks if all initialization keyword arguments are non-empty strings.

        Returns:
            bool: True if all initialization keyword arguments are non-empty strings, False otherwise.
        """
        for arg_name, arg_val in self.init_kwargs.items():
            if arg_val == '':
                return False
        return True
    
    def all_call_kwargs_non_empty_strings(self):
        """
        Checks if all call keyword arguments are non-empty strings.

        Returns:
            bool: True if all call keyword arguments are non-empty strings, False otherwise.
        """
        for arg_name, arg_val in self.call_kwargs.items():
            if arg_val == '':
                return False
        return True
    
    def can_potentially_run(self):
        """
        Determines if the step can potentially be executed based on argument validation.
        Very simply checks if all __init__ and __call__  keyword arguments for the object are
        non-empty strings.
        
        Returns:
            bool: True if the step can potentially run, False otherwise.
        """
        return self.all_init_kwargs_non_empty_strings() and self.all_call_kwargs_non_empty_strings()

