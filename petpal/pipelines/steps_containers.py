import copy
from typing import Union
import networkx as nx
from matplotlib import pyplot as plt
from .steps_base import *
from .preproc_steps import PreprocStepType, ImageToImageStep, TACsFromSegmentationStep, ResampleBloodTACStep
from .kinetic_modeling_steps import KMStepType, GraphicalAnalysisStep, TCMFittingAnalysisStep, ParametricGraphicalAnalysisStep

StepType = Union[FunctionBasedStep, ObjectBasedStep, PreprocStepType, KMStepType]

class StepsContainer:
    """
    A container for managing and executing a sequence of steps in a pipeline.

    This class allows for the addition, removal, and execution of steps, as well as printing their
    details and combining multiple step containers.

    Attributes:
        name (str): Name of the steps container.
        step_objs (list[StepType]): List of step objects in the container.
        step_names (list[str]): List of step names in the container.
    """
    def __init__(self, name: str, *steps: StepType):
        """
        Initializes the StepsContainer with a name and an optional sequence of steps.

        Args:
            name (str): Name of the steps container.
            *steps (StepType): Optional sequence of steps to add to the container.
        """
        self.name = name
        self.step_objs: list[StepType] = []
        self.step_names: list[str] = []
        for step in steps:
            self.add_step(step)
    
    def __repr__(self):
        """
        Provides an unambiguous string representation of the TACsFromSegmentationStep instance.

        Returns:
            str: A string representation showing how the instance can be recreated.
        """
        cls_name = type(self).__name__
        info_str = [f'{cls_name}(', f'{repr(self.name)},']
        
        for step_obj in self.step_objs:
            info_str.append(f'{repr(step_obj)},')
        
        info_str.append(')')
        
        return f'\n    '.join(info_str)
    
    def add_step(self, step: StepType):
        """
        Adds a step to the container if it is not already present.

        Args:
            step (StepType): The step to add.

        Raises:
            TypeError: If the step is not of the correct type.
            KeyError: If a step with the same name already exists.
        """
        if not isinstance(step, StepType.__args__):
            raise TypeError("Step must be of type StepType")
        
        if step.name not in self.step_names:
            self.step_objs.append(copy.deepcopy(step))
            self.step_names.append(self.step_objs[-1].name)
        else:
            raise KeyError("A step with this name already exists.")
    
    def remove_step(self, step: Union[int, str]):
        """
        Removes a step from the container by index or name.

        Args:
            step (Union[int, str]): The index or name of the step to remove.

        Raises:
            IndexError: If the step index does not exist.
            KeyError: If the step name does not exist.
        """
        if isinstance(step, int):
            try:
                del self.step_objs[step]
                del self.step_names[step]
            except IndexError:
                raise IndexError(f"Step number {step} does not exist.")
        elif isinstance(step, str):
            if step not in self.step_names:
                raise KeyError(f"Step name {step} does not exist.")
            try:
                step_index = self.step_names.index(step)
                del self.step_objs[step_index]
                del self.step_names[step_index]
            except Exception:
                raise Exception
    
    def print_step_details(self):
        """
        Prints the details of all steps in the container.
        """
        print(f"({self.name} StepsContainer Info):")
        if not self.step_objs:
            print("No steps in container.")
        else:
            print("*" * 80)
            for step_id, a_step in enumerate(self.step_objs):
                print('-' * 70)
                print(f"Step Number {step_id + 1:>02}")
                print("  "+"    \n".join(str(a_step).split('\n')))
                print('-' * 70)
            print("*" * 80)
    
    def __str__(self) -> str:
        """
        Returns a detailed string representation of the StepsContainer instance.

        Returns:
            str: A string describing the steps-container, including its name, and the
            names of the sequence of steps.
        """
        info_str = [f"({self.name} steps-container info):"]
        if not self.step_objs:
            info_str.append('No steps in container.')
        else:
            info_str.append('-' * 70)
            for step_id, step_name in enumerate(self.step_names):
                info_str.append(f"  Step number {step_id + 1:>02}: {step_name}")
            info_str.append('-' * 70)
        return "\n".join(info_str)
    
    def __call__(self):
        """
        Executes all steps in the container in sequence.
        """
        for step_id, (step_name, a_step) in enumerate(zip(self.step_names, self.step_objs)):
            a_step.execute()
    
    def __getitem__(self, step: Union[int, str]):
        """
        Gets a step from the container by index or name.

        Args:
            step (Union[int, str]): The index or name of the step to get.

        Returns:
            StepType: The requested step.

        Raises:
            IndexError: If the step index does not exist.
            KeyError: If the step name does not exist.
            TypeError: If the key is not an integer or string.
        """
        if isinstance(step, int):
            try:
                return self.step_objs[step]
            except IndexError:
                raise IndexError(f"Step number {step} does not exist.")
        elif isinstance(step, str):
            if step not in self.step_names:
                raise KeyError(f"Step name {step} does not exist.")
            try:
                step_index = self.step_names.index(step)
                return self.step_objs[step_index]
            except KeyError:
                raise KeyError(f"Step name {step} does not exist.")
        else:
            raise TypeError(f"Key must be an integer or a string. Got {type(step)}")
    
    def __add__(self, other: 'StepsContainer') -> 'StepsContainer':
        """
        Combines this StepsContainer with another one. The other container cannot have steps with the same name.

        Args:
            other (StepsContainer): The other steps container to combine with.

        Returns:
            StepsContainer: A new StepsContainer containing steps from both containers.

        Raises:
            TypeError: If the other object is not a StepsContainer.
            KeyError: If a step with the same name already exists.
        """
        if isinstance(other, StepsContainer):
            new_container_name = f"{self.name}-{other.name}"
            new_container = StepsContainer(new_container_name)
            for step in self.step_objs:
                new_container.add_step(step)
            for step in other.step_objs:
                new_container.add_step(step)
            return new_container
        else:
            raise TypeError("Can only add another StepsContainer instance")
    
    @classmethod
    def default_preprocess_steps(cls, name: str = 'preproc'):
        """
        Creates a default StepsContainer with common preprocessing steps.
        
        We have the following steps in sequence:
            - :meth:`Threshold Based Cropping<petpal.pipelines.preproc_steps.ImageToImageStep.default_threshold_cropping>`.
            - :meth:`Motion correct frames brighter than mean-image<petpal.pipelines.preproc_steps.ImageToImageStep.default_moco_frames_above_mean>`.
            - :meth:`Register PET to Anatomical<petpal.pipelines.preproc_steps.ImageToImageStep.default_register_pet_to_t1>`.
            - :meth:`Write ROI TACs from segmentation<petpal.pipelines.preproc_steps.TACsFromSegmentationStep.default_write_tacs_from_segmentation_rois>`.
            - :meth:`Resample blood on scanner frame times<petpal.pipelines.preproc_steps.ResampleBloodTACStep.default_resample_blood_tac_on_scanner_times>`.

        Args:
            name (str, optional): Name of the steps container. Defaults to 'preproc'.

        Returns:
            StepsContainer: A new StepsContainer with default preprocessing steps.
        """
        obj = cls(name=name)
        obj.add_step(ImageToImageStep.default_threshold_cropping())
        obj.add_step(ImageToImageStep.default_moco_frames_above_mean())
        obj.add_step(ImageToImageStep.default_register_pet_to_t1())
        obj.add_step(TACsFromSegmentationStep.default_write_tacs_from_segmentation_rois())
        obj.add_step(ResampleBloodTACStep.default_resample_blood_tac_on_scanner_times())
        return obj
    
    @classmethod
    def default_graphical_analysis_steps(cls, name: str = 'km_graphical_analysis'):
        """
        Creates a default StepsContainer with common graphical analysis steps.
        
        We have the following steps in sequence:
            - :meth:`Patlak<petpal.pipelines.kinetic_modeling_steps.GraphicalAnalysisStep.default_patlak>`.
            - :meth:`Logan<petpal.pipelines.kinetic_modeling_steps.GraphicalAnalysisStep.default_logan>`.
            - :meth:`Alt-Logan<petpal.pipelines.kinetic_modeling_steps.GraphicalAnalysisStep.default_alt_logan>`.

        Args:
            name (str, optional): Name of the steps container. Defaults to 'km_graphical_analysis'.

        Returns:
            StepsContainer: A new StepsContainer with default graphical analysis steps.
            
        Notes:
            The steps do not technically depend on each other and can be run out of sequence.
        """
        obj = cls(name=name)
        obj.add_step(GraphicalAnalysisStep.default_patlak())
        obj.add_step(GraphicalAnalysisStep.default_logan())
        obj.add_step(GraphicalAnalysisStep.default_alt_logan())
        return obj
    
    @classmethod
    def default_parametric_graphical_analysis_steps(cls, name: str = 'km_parametric_graphical_analysis'):
        """
        Creates a default StepsContainer with common parametric graphical analysis steps.
        
        We have the following steps in sequence:
            - :meth:`Patlak<petpal.pipelines.kinetic_modeling_steps.ParametricGraphicalAnalysisStep.default_patlak>`.
            - :meth:`Logan<petpal.pipelines.kinetic_modeling_steps.ParametricGraphicalAnalysisStep.default_logan>`.
            - :meth:`Alt-Logan<petpal.pipelines.kinetic_modeling_steps.ParametricGraphicalAnalysisStep.default_alt_logan>`.

        Args:
            name (str, optional): Name of the steps container. Defaults to 'km_parametric_graphical_analysis'.

        Returns:
            StepsContainer: A new StepsContainer with default parametric graphical analysis steps.
            
        Notes:
            The steps do not technically depend on each other and can be run out of sequence.
        """
        obj = cls(name=name)
        obj.add_step(ParametricGraphicalAnalysisStep.default_patlak())
        obj.add_step(ParametricGraphicalAnalysisStep.default_logan())
        obj.add_step(ParametricGraphicalAnalysisStep.default_alt_logan())
        return obj
    
    @classmethod
    def default_tcm_analysis_steps(cls, name: str = 'km_tcm_analysis'):
        """
        Creates a default StepsContainer with common TCM analysis steps.
        
        We have the following steps in sequence:
            - :meth:`1TCM<petpal.pipelines.kinetic_modeling_steps.TCMFittingAnalysisStep.default_1tcm>`.
            - :meth:`Serial 2TCM<petpal.pipelines.kinetic_modeling_steps.TCMFittingAnalysisStep.default_serial2tcm>`.
            - :meth:`Irreversible 2TCM<petpal.pipelines.kinetic_modeling_steps.TCMFittingAnalysisStep.default_irreversible_2tcm>`.

        Args:
            name (str, optional): Name of the steps container. Defaults to 'km_tcm_analysis'.

        Returns:
            StepsContainer: A new StepsContainer with default TCM analysis steps.
        """
        obj = cls(name=name)
        obj.add_step(TCMFittingAnalysisStep.default_1tcm())
        obj.add_step(TCMFittingAnalysisStep.default_serial2tcm())
        obj.add_step(TCMFittingAnalysisStep.default_irreversible_2tcm())
        return obj
    
    @classmethod
    def default_kinetic_analysis_steps(cls, name: str = 'km'):
        """
        Creates a default StepsContainer with common kinetic analysis steps.
        
        We have the following steps in sequence:
            - :meth:`ROI TACs: Patlak<petpal.pipelines.kinetic_modeling_steps.GraphicalAnalysisStep.default_patlak>`.
            - :meth:`ROI TACs: Logan<petpal.pipelines.kinetic_modeling_steps.GraphicalAnalysisStep.default_logan>`.
            - :meth:`ROI TACs: Alt-Logan<petpal.pipelines.kinetic_modeling_steps.GraphicalAnalysisStep.default_alt_logan>`.
            - :meth:`Parametric: Patlak<petpal.pipelines.kinetic_modeling_steps.ParametricGraphicalAnalysisStep.default_patlak>`.
            - :meth:`Parametric: Logan<petpal.pipelines.kinetic_modeling_steps.ParametricGraphicalAnalysisStep.default_logan>`.
            - :meth:`Parametric: Alt-Logan<petpal.pipelines.kinetic_modeling_steps.ParametricGraphicalAnalysisStep.default_alt_logan>`.
            - :meth:`ROI TACs: 1TCM<petpal.pipelines.kinetic_modeling_steps.TCMFittingAnalysisStep.default_1tcm>`.
            - :meth:`ROI TACs: Serial 2TCM<petpal.pipelines.kinetic_modeling_steps.TCMFittingAnalysisStep.default_serial2tcm>`.
            - :meth:`ROI TACs: Irreversible 2TCM<petpal.pipelines.kinetic_modeling_steps.TCMFittingAnalysisStep.default_irreversible_2tcm>`.

        Args:
            name (str, optional): Name of the steps container. Defaults to 'km'.

        Returns:
            StepsContainer: A new StepsContainer with default kinetic analysis steps.
            
        Notes:
            The steps do not technically depend on each other and can be run out of sequence.
        """
        
        parametric_graphical_analysis_steps = cls.default_parametric_graphical_analysis_steps()
        graphical_analysis_steps = cls.default_graphical_analysis_steps()
        tcm_analysis_steps = cls.default_tcm_analysis_steps()
        
        obj = parametric_graphical_analysis_steps + graphical_analysis_steps + tcm_analysis_steps
        obj.name = name
        return obj


class StepsPipeline:
    """
    A pipeline for managing and executing a sequence of steps organized in containers, with support for dependencies.

    This class allows for the addition, removal, and execution of step containers,
    as well as managing dependencies between the steps.

    Attributes:
        name (str): Name of the steps pipeline.
        step_containers (dict[str, StepsContainer]): Dictionary of step containers in the pipeline.
        dependency_graph (nx.DiGraph): Directed acyclic graph representing dependencies between steps.
        
    See Also:
        :class:`StepsContainer`
    """
    def __init__(self, name: str, step_containers: list[StepsContainer]):
        """
        Initializes the StepsPipeline with a name and an optional list of step containers.

        Args:
            name (str): Name of the steps pipeline.
            step_containers (list[StepsContainer]): Optional list of step containers to add to the pipeline.
        """
        self.name: str = name
        self.step_containers: dict[str, StepsContainer] = {}
        self.dependency_graph = nx.DiGraph()
        
        for container in step_containers:
            self.add_container(container)
    
    def __repr__(self):
        """
        Provides an unambiguous string representation of the TACsFromSegmentationStep instance.

        .. important::
            This ``repr`` does not include the dependency graph.

        Returns:
            str: A string representation showing how the instance can be recreated.
        """
        cls_name = type(self).__name__
        info_str = [f'{cls_name}(', f'name={repr(self.name)},' 'step_containers=[']
        
        for _, container_obj in self.step_containers.items():
            info_str.append(f'{repr(container_obj)},')
        info_str.append(']')
        info_str.append(')')
        
        return f'\n    '.join(info_str)
    
    def __str__(self) -> str:
        """
        Returns a detailed string representation of the StepsPipeline instance.

        Returns:
            str: A string describing the steps-pipeline, including its name, the steps in
            each of the steps-containers, and the output dependencies of each step.
        """
        sep_line_len = 50
        info_str = [f'({self.name} pipeline info):', "*" * sep_line_len]
        for con_name, con_obj in self.step_containers.items():
            info_str.append(f"  ({con_name} steps-container info):")
            for step_num, step_name in enumerate(con_obj.step_names):
                info_str.append(f"    Step {step_num:>02}: {repr(step_name)}")
                step_deps = list(self.dependency_graph[step_name].keys())
                if step_deps:
                    info_str.append(fr"      \> sends output to:")
                    for dep_name in step_deps[:-1]:
                        info_str.append(fr"            ├── {repr(dep_name)}")
                    for dep_name in step_deps[-1:]:
                        info_str.append(fr"            └── {repr(dep_name)}")
            info_str.append("-" * sep_line_len)
        info_str.append("*" * sep_line_len)
        return "\n".join(info_str)
    
    def add_container(self, step_container: StepsContainer):
        """
        Adds a step container to the pipeline.

        Args:
            step_container (StepsContainer): The step container to add.

        Raises:
            TypeError: If the step container is not an instance of StepsContainer.
            KeyError: If a container with the same name already exists.
        """
        if not isinstance(step_container, StepsContainer):
            raise TypeError("`step_container` must be an instance of StepsContainer")
        
        container_name = step_container.name
        if container_name in self.step_containers.keys():
            raise KeyError(f"Container name {container_name} already exists.")
        
        self.step_containers[container_name] = copy.deepcopy(step_container)
        
        step_objs = self.step_containers[container_name].step_objs
        for step in step_objs:
            self.dependency_graph.add_node(f"{step.name}", grp=container_name)
    
    def __call__(self):
        """
        Executes all steps in the pipeline in topologically sorted order.
        """
        for step_name in nx.topological_sort(self.dependency_graph):
            step = self.get_step_from_node_label(node_label=step_name)
            step.execute()
    
    def add_step(self, container_name: str, step: StepType):
        """
        Adds a step to a specified container in the pipeline.

        Args:
            container_name (str): The name of the container to add the step to.
            step (StepType): The step to add.

        Raises:
            KeyError: If the container or step name does not exist or if the step name already exists.
        """
        if container_name not in self.step_containers.keys():
            raise KeyError(f"Container name {container_name} does not exist.")
        if step.name in self.dependency_graph:
            raise KeyError(f"Step name {step.name} already exists. Steps must have unique names.")
        
        self.step_containers[container_name].add_step(step)
        self.dependency_graph.add_node(f"{step.name}", grp=container_name)
    
    def remove_step(self, step: str):
        """
        Removes a step from the pipeline based on its name.

        Args:
            step (str): The name of the step to remove.

        Raises:
            KeyError: If the step name does not exist.
        """
        node_names = list(self.dependency_graph.nodes)
        if step not in node_names:
            raise KeyError(f"Step name {step} does not exist.")
        graph_nodes = self.dependency_graph.nodes(data=True)
        container_name = graph_nodes[step]['grp']
        self.dependency_graph.remove_node(step)
        self.step_containers[container_name].remove_step(step)
    
    def print_steps_names(self, container_name: Union[str, None] = None):
        """
        Prints the names of all steps in the pipeline or a specific container.

        Args:
            container_name (Union[str, None], optional): The name of the container to print step names from.
                If None, prints step names from all containers.

        Raises:
            KeyError: If the container name does not exist.
        """
        if container_name is None:
            for name, container in self.step_containers.items():
                print(str(container))
        elif container_name in self.step_containers.keys():
            print(str(self.step_containers[container_name]))
        else:
            raise KeyError(f"Container name {container_name} does not exist. ")
    
    def print_steps_details(self, container_name: Union[str, None] = None):
        """
        Prints the details of all steps in the pipeline or a specific container.

        Args:
            container_name (Union[str, None], optional): The name of the container to print step details from.
                If None, prints step details from all containers.

        Raises:
            KeyError: If the container name does not exist.
        """
        if container_name is None:
            for name, container in self.step_containers.items():
                container.print_step_details()
        elif container_name in self.step_containers.keys():
            self.step_containers[container_name].print_step_details()
        else:
            raise KeyError(f"Container name {container_name} does not exist. ")
    
    def add_dependency(self, sending: str, receiving: str):
        """
        Adds a dependency between two steps in the pipeline.

        Args:
            sending (str): The name of the sending step.
            receiving (str): The name of the receiving step.

        Raises:
            KeyError: If either step name does not exist.
            RuntimeError: If adding the dependency creates a cycle in the dependency graph.
        """
        node_names = list(self.dependency_graph.nodes)
        if sending not in node_names:
            raise KeyError(f"Step name {sending} does not exist.")
        if receiving not in node_names:
            raise KeyError(f"Step name {receiving} does not exist.")
        
        self.dependency_graph.add_edge(sending, receiving)
        
        if not nx.is_directed_acyclic_graph(self.dependency_graph):
            raise RuntimeError(f"Adding dependency {sending} -> {receiving} creates a cycle!")
    
    def get_step_from_node_label(self, node_label: str) -> StepType:
        """
        Retrieves a step object based on its node label in the dependency graph.

        Args:
            node_label (str): The label of the node representing the step.

        Returns:
            StepType: The step object.

        Raises:
            KeyError: If the step name does not exist in the graph.
        """
        if node_label not in self.dependency_graph.nodes:
            raise KeyError(f"Step name {node_label} does not exist.")
        graph_nodes = self.dependency_graph.nodes(data=True)
        container_name = graph_nodes[node_label]['grp']
        return self.step_containers[container_name][node_label]
    
    def update_dependencies_for(self, step_name: str, verbose=False):
        """
        Updates the dependencies for a specified step, setting inputs from the outputs of its dependencies.

        Args:
            step_name (str): The name of the step to update dependencies for.
            verbose (bool, optional): If True, prints detailed information during the update process. Defaults to False.
        """
        sending_step = self.get_step_from_node_label(step_name)
        for an_edge in self.dependency_graph[step_name]:
            receiving_step = self.get_step_from_node_label(an_edge)
            try:
                receiving_step.set_input_as_output_from(sending_step)
            except NotImplementedError:
                if verbose:
                    print(f"Step `{receiving_step.name}` of type `{type(receiving_step).__name__}` does not have a "
                          f"set_input_as_output_from method implemented.\nSkipping.")
            else:
                if verbose:
                    print(f"Updated input-output dependency between {sending_step.name} and {receiving_step.name}")
    
    def update_dependencies(self, verbose=False):
        """
        Updates the dependencies for all steps in the pipeline.

        Args:
            verbose (bool, optional): If True, prints detailed information during the update process. Defaults to False.
        """
        for step_name in nx.topological_sort(self.dependency_graph):
            self.update_dependencies_for(step_name=step_name, verbose=verbose)
    
    def get_steps_potential_run_state(self) -> dict:
        """
        Gets a dictionary representing the potential run state of all steps in topologically sorted order.

        Returns:
            dict: Dictionary mapping step names to their potential run state (True if the step can potentially run, False otherwise).
        """
        step_maybe_runnable = {}
        for node_name in nx.topological_sort(self.dependency_graph):
            step = self.get_step_from_node_label(node_name)
            step_maybe_runnable[node_name] = step.can_potentially_run()
        return step_maybe_runnable
    
    def can_steps_potentially_run(self) -> bool:
        """
        Checks if all steps in the pipeline can potentially run.

        Returns:
            bool: True if all steps can potentially run, False otherwise.
        """
        steps_run_state = self.get_steps_potential_run_state()
        for step_name, run_state in steps_run_state.items():
            if not run_state:
                return False
        return True
    
    def plot_dependency_graph(self,
                              figsize: tuple = (12, 12),
                              node_options: Union[dict, None] = None,
                              label_options: Union[dict, None] = None,
                              draw_options: Union[dict, None] = None) -> None:
        """
        Plots the dependency graph of the steps pipeline.

        Args:
            figsize (tuple, optional): Figure size for the plot. Defaults to (12, 12).
            node_options (dict, optional): Additional options for drawing nodes. Defaults to None.
            label_options (dict, optional): Additional options for node labels. Defaults to None.
            draw_options (dict, optional): Additional options for the draw_networkx function. Defaults to None.
        
        Notes:
            We have the following node options.
            
            
            .. code-block:: python
            
                default_node_options = {'node_size': 2000, 'node_color': 'w', 'node_shape': 'o'}
        
                default_label_options = {'verticalalignment': 'center',
                                         'clip_on': False,
                                         'font_size': 15,
                                         'bbox' : dict(facecolor='azure', edgecolor='blue',
                                                       boxstyle='round', alpha=0.25, linewidth=2)
                                        }
                
                default_draw_options = {'with_labels': True, 'arrows': True, 'arrowsize': 30, 'width': 2}
            
        """
        dep_graph = self.dependency_graph
        
        for layer, nodes in enumerate(nx.topological_generations(dep_graph)):
            for node in nodes:
                dep_graph.nodes[node]['layer'] = layer
        
        dep_pos = nx.multipartite_layout(dep_graph, subset_key='layer', align='horizontal')
        dep_labels = {a_lab: '\n'.join(a_lab.split("_")) for a_lab in dep_graph.nodes}
        
        default_node_options = {'node_size': 2000, 'node_color': 'w', 'node_shape': 'o'}
        
        default_label_options = {'verticalalignment': 'center',
                                 'clip_on': False,
                                 'font_size': 15,
                                 'bbox' : dict(facecolor='azure', edgecolor='blue',
                                               boxstyle='round', alpha=0.25, linewidth=2)
                                }
        
        default_draw_options = {'with_labels': True, 'arrows': True, 'arrowsize': 30, 'width': 2}
        
        if node_options:
            default_node_options = {**default_node_options, **node_options}
        if label_options:
            default_label_options = {**default_label_options, **label_options}
        if draw_options:
            default_draw_options = {**default_draw_options, **draw_options}
        
        myFig, myAx = plt.subplots(figsize=figsize, constrained_layout=True)
        nx.draw_networkx(dep_graph, ax=myAx, pos=dep_pos, labels=dep_labels,
                         **default_node_options,
                         **default_label_options,
                         **default_draw_options)
        
        plt.show()
    
    def print_dependency_graph(self):
        """
        Prints a textual representation of the dependency graph for the steps pipeline.

        This method outputs the steps in the pipeline in topologically sorted order,
        showing their dependencies. For each step, it lists the subsequent steps
        that depend on it, formatted in a tree-like structure.
        """
        sep_line_len = 50
        info_str = [f'({self.name} pipeline dependency info):', "*" * sep_line_len]
        for step_name in nx.topological_sort(self.dependency_graph):
            info_str.append(f'{repr(step_name)}')
            step_deps = list(self.dependency_graph[step_name].keys())
            if step_deps:
                info_str[-1] = f"{info_str[-1]} sends output to"
                for dep_name in step_deps[:-1]:
                    info_str.append(f"  ├── {repr(dep_name)}")
                for dep_name in step_deps[-1:]:
                    info_str.append(f"  └── {repr(dep_name)}")
            else:
                info_str[-1] = f"{info_str[-1]} has no output dependencies"
        
        info_str.append('*' * sep_line_len)
        print('\n'.join(info_str))
        
    
    @classmethod
    def default_steps_pipeline(cls, name='PET-MR_analysis'):
        """
        Creates a default steps pipeline with predefined containers and dependencies.
        
        Contains the following step-containers:
            - :meth:`Pre-processing<StepsContainer.default_preprocess_steps>`.
            - :meth:`Kinetic modeling<StepsContainer.default_kinetic_analysis_steps>`.
            
        The pipeline has the following steps dependency:
        
        .. plot::
            :include-source:
            :caption: The dependency graph of the steps pipeline. From raw PET `.nii` images we crop the images
                in each dimension based on a threshold, perform motion correction for each frame where the per-frame
                mean-intensity is larger than the overall time-averaged intensity of the PET series, register the
                PET series to anatomical space, extract all the TACs corresponding the ROIs defined via segmentation,
                and resample the blood TAC onto the scanner frame times as part of pre-processing. Then, we perform
                ROI-based graphical and full tissue compartment model fits, and parametric graphical analysis fits.
                
            from petpal.pipelines.steps_containers import StepsPipeline
            
            # Instantiate the default PET/MR pipeline
            my_pipe = StepsPipeline.default_steps_pipeline()
            
            # Plot the dependency graph of the steps pipeline
            my_pipe.plot_dependency_graph()
        

        Args:
            name (str, optional): Name of the steps pipeline. Defaults to 'PET-MR_analysis'.

        Returns:
            StepsPipeline: A new StepsPipeline instance with default setup.
        """
        obj = cls(name=name, step_containers=[StepsContainer.default_preprocess_steps(name='preproc'),
                                              StepsContainer.default_kinetic_analysis_steps(name='km')])
        
        obj.add_dependency(sending='thresh_crop', receiving='moco_frames_above_mean')
        obj.add_dependency(sending='moco_frames_above_mean', receiving='register_pet_to_t1')
        obj.add_dependency(sending='register_pet_to_t1', receiving='write_roi_tacs')
        obj.add_dependency(sending='register_pet_to_t1', receiving='resample_PTAC_on_scanner')
        
        for method in ['patlak', 'logan', 'alt_logan']:
            obj.add_dependency(sending='register_pet_to_t1', receiving=f'parametric_{method}_fit')
            obj.add_dependency(sending='resample_PTAC_on_scanner', receiving=f'parametric_{method}_fit')
        
        for fit_model in ['1tcm', '2tcm-k4zero', 'serial-2tcm', 'patlak', 'logan', 'alt_logan']:
            obj.add_dependency(sending='write_roi_tacs', receiving=f"roi_{fit_model}_fit")
            obj.add_dependency(sending='resample_PTAC_on_scanner', receiving=f"roi_{fit_model}_fit")
        
        return obj