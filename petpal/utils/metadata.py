"""Utilities for metadata handling, scrubbing, etc..."""

from shutil import copyfile
from itertools import accumulate

from .image_io import safe_load_meta, write_dict_to_json
from .constants import HALF_LIVES
from .scan_timing import calculate_frame_reference_time
from ..preproc.decay_correction import calculate_frame_decay_factor


class BidsMetadataMender:
    """Class for repairing and filling in the gaps of BIDS metadata based on existing fields."""

    metadata: dict
    filepath: str
    decay_correction: bool

    def __init__(self, json_filepath: str, decay_correction: bool = False):
        self.metadata = safe_load_meta(input_metadata_file=json_filepath)
        self.filepath = json_filepath
        self.decay_correction = decay_correction


    def __call__(self, output_filepath : str | None = None):
        self._add_missing_keys()
        self._to_file(output_filepath)
        

    def _add_missing_keys(self):
        updated_keys = []
        if 'FrameDuration' in self.metadata:
            self._add_frame_times_start()
            updated_keys.append('FrameTimesStart')
        if 'TracerRadionuclide' in self.metadata:
            self._add_half_life()
            updated_keys.append('RadionuclideHalfLife')
        if {'RadionuclideHalfLife', 'FrameDuration', 'FrameTimesStart'}.issubset(self.metadata):
            self._add_frame_reference_times()
            updated_keys.append('FrameReferenceTime')
        if self.decay_correction and {'RadionuclideHalfLife', 'FrameReferenceTime'}.issubset(self.metadata):
            self._add_decay_factors()
            updated_keys += ['DecayCorrectionFactor','ImageDecayCorrected']
        else: 
            self._add_empty_decay_factors()
            updated_keys += ['DecayCorrectionFactor','ImageDecayCorrected']
        print(f'The following keys were updated: {updated_keys}')


    def _add_half_life(self):
        """Add "RadionuclideHalfLife" key to metadata."""
        metadata = self.metadata
        radionuclide = metadata['TracerRadionuclide'].lower().replace("-", "")
        half_life = float(HALF_LIVES[radionuclide])
        metadata['RadionuclideHalfLife'] = half_life
        self.metadata = metadata


    def _add_empty_decay_factors(self):
        """Adds a list of ones for decay factors and sets 'ImageDecayCorrected' to False."""
        metadata = self.metadata
        frame_durations = metadata['FrameDuration']
        decay_factors = [1 for i in frame_durations]
        metadata['DecayCorrectionFactor'] = decay_factors
        metadata['ImageDecayCorrected'] = 'False'
        self.metadata = metadata


    def _add_decay_factors(self):
        """Computes decay factors and adds 'DecayCorrectionFactor' to metadata."""
        metadata = self.metadata
        half_life = metadata['RadionuclideHalfLife']
        decay_factors = [calculate_frame_decay_factor(frame_reference_time=t, half_life=half_life) for t in metadata['FrameReferenceTime']]
        metadata['DecayCorrectionFactor'] = decay_factors
        metadata.pop('DecayFactor', None)
        metadata['ImageDecayCorrected'] = 'True'
        self.metadata = metadata
        

    def _add_frame_times_start(self):
        """Fill in frame starts from frame durations, assuming first frame starts at 0."""
        metadata = self.metadata
        frame_durations = metadata['FrameDuration']
        frame_starts = [0]
        frame_starts = frame_starts + list(accumulate(frame_durations[:-1]))
        metadata['FrameTimesStart'] = frame_starts
        self.metadata = metadata


    def _add_frame_reference_times(self):
        """Fill in frame reference times from frame starts and durations."""
        metadata = self.metadata
        half_life = metadata['RadionuclideHalfLife']
        frame_starts = metadata['FrameTimesStart']
        frame_durations = metadata['FrameDuration']
        frame_reference_times = [calculate_frame_reference_time(frame_duration=duration,frame_start=start,half_life=half_life) for start, duration in zip(frame_starts, frame_durations)]
        metadata['FrameReferenceTime'] = frame_reference_times
        self.metadata = metadata


    def _to_file(self, filepath : str | None = None):
        """Write metadata dictionary to a .json file; defaults to making a backup file *.bak before overwriting the initial .json."""
        if filepath is None: 
            filepath = self.filepath
            copyfile(src=filepath, dst=filepath+".bak")
        write_dict_to_json(meta_data_dict=self.metadata, out_path=filepath)
