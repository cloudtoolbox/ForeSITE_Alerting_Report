import platform
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import rpy2.robjects as robjects
from pandas.tseries import offsets
from rpy2.robjects import numpy2ri, pandas2ri, r
from rpy2.robjects.packages import importr

from epysurv.metrics.outbreak_detection import ghozzi_score


def silence_r_output():
    """Silence output from R code.

    This is useful, because some algorithm otherwise print every time they are invoked.
    """
    if platform.system() == "Linux":
        r.sink("/dev/null")
    elif platform.system() == "Windows":
        r.sink("NUL")


silence_r_output()
if all(
    hasattr(module, "converter") for module in (numpy2ri, pandas2ri)
) and hasattr(robjects, "default_converter"):
    # rpy2 >= 3.5: keep default conversion to avoid deprecated activate().
    pass
else:
    numpy2ri.activate()
    pandas2ri.activate()
surveillance = importr("surveillance")


@dataclass
class TimepointSurveillanceAlgorithm:
    """Algorithms that predict outbreaks for every timepoint."""

    _training_data: pd.DataFrame = field(init=False, repr=False)

    def fit(self, data: pd.DataFrame) -> "TimepointSurveillanceAlgorithm":
        """Expects data with time series index, case counts and outbreak labels."""
        self._validate_data(data)
        data = data.copy()
        if "n_outbreak_cases" in data.columns:
            # Remove outbreaks cases from baseline.
            data["n_cases"] -= data["n_outbreak_cases"]
        else:
            warnings.warn(
                'The column "n_outbreak_cases" is not present in input parameter `data`. '
                '"n_cases" is treated as if it contains no outbreaks.'
            )

        self._training_data = data
        return self

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """Expects data with time series index and case counts."""
        self._data_in_the_future(data)

    def score(self, data_with_labels: pd.DataFrame):
        prediction_result = self.predict(data_with_labels)
        return ghozzi_score(prediction_result)

    def _validate_data(self, data: pd.DataFrame):
        self._contains_dates(data)
        self._contains_counts(data)

    def _contains_dates(self, data: pd.DataFrame):
        has_dates = isinstance(data.index, pd.DatetimeIndex)
        if not has_dates:
            raise ValueError("`data` needs to have a datetime index.")

    def _contains_counts(self, data: pd.DataFrame):
        if "n_cases" not in data.columns:
            raise ValueError('No column named "n_cases" in `data`')

    def _data_in_the_future(self, data: pd.DataFrame):
        if data.index.min() <= self._training_data.index.max():
            raise ValueError("The prediction data overlaps with the training data.")


offset_to_freq = {
    offsets.Day: 365,
    offsets.Week: 52,
    offsets.MonthBegin: 12,
    offsets.MonthEnd: 12,
}

offset_to_attr = {
    offsets.Day: "day",
    offsets.Week: "week",
    offsets.MonthBegin: "month",
    offsets.MonthEnd: "month",
}


def _get_freq(data) -> int:
    return offset_to_freq[type(data.index.freq)]


def _get_start_epoch(data: pd.DataFrame) -> int:
    return getattr(data.index[0], offset_to_attr[type(data.index.freq)])


class SurveillanceRPackageAlgorithm(TimepointSurveillanceAlgorithm):
    """Base class for the algorithm from the R package surveillance."""

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Predict outbreaks.

        Parameters
        ----------
        data
            Dataframe with DateTimeIndex containing the columns "n_cases".

        Returns
        -------
            Original dataframe with "alarm" column and other relevant columns as available (e.g. "upperbound") added.
        """
        super().predict(data)
        # Concat training and prediction data. Make index array for range param.
        full_data = (
            pd.concat((self._training_data, data), keys=["train", "test"], sort=True)
            .reset_index(level=0)
            .rename(columns={"level_0": "provenance"})
        )
        r_instance = self._prepare_r_instance(full_data)
        # R indexes are 1-based. Therefore we add 1.
        detection_range = robjects.IntVector(
            [int(i) for i in (np.where(full_data.provenance == "test")[0] + 1)]
        )
        surveillance_result = self._call_surveillance_algo(r_instance, detection_range)
        data = data.assign(
            alarm=self._extract_slot(surveillance_result, "alarm").astype(bool)
        )

        # Let's check what other slots were returned
        slot_keys = set()
        if hasattr(surveillance_result, "slotnames"):
            slot_keys = set(surveillance_result.slotnames())
        elif hasattr(surveillance_result, "names"):
            names = surveillance_result.names
            slot_keys = set(names() if callable(names) else names)

        if "upperbound" in slot_keys:
            data = data.assign(
                upperbound=self._extract_slot(surveillance_result, "upperbound").astype(
                    float
                )
            )

        return data

    def _None_to_NULL(self, obj):  # NOQA
        return robjects.NULL if obj is None else obj

    def _prepare_r_instance(self, data: pd.DataFrame):
        """Transform dataframe into R data structure on which the R algorithm can work."""
        raise NotImplementedError

    def _extract_slot(self, surveillance_result, slot_name) -> np.ndarray:
        """Extract the array for the requested slot name from the surveillance result R data structure."""
        raise NotImplementedError

    def _call_surveillance_algo(self, sts, detection_range) -> pd.DataFrame:
        raise NotImplementedError


class STSBasedAlgorithm(SurveillanceRPackageAlgorithm):
    """Base class for algorithms that operate on the STS (SurveillanceTimeSeries) class."""

    def _prepare_r_instance(self, data: pd.DataFrame):
        if data.index.freq is None:
            freq = pd.infer_freq(data.index)
            if freq is None:
                raise ValueError(
                    f"The time series index has no valid frequency. Index={data.index}"
                )
            data.index.freq = freq

        epoch_days = (
            (data.index.normalize() - pd.Timestamp("1970-01-01")) / pd.Timedelta(days=1)
        ).astype(float)
        sts = surveillance.sts(
            start=robjects.IntVector(
                [int(data.index[0].year), int(_get_start_epoch(data))]
            ),
            # Keep epoch as numeric days since 1970-01-01.
            # Passing as.Date() can produce an R "array" class in some environments,
            # while sts expects the slot to be numeric.
            epoch=robjects.FloatVector(epoch_days.tolist()),
            freq=_get_freq(data),
            observed=robjects.IntVector(data["n_cases"].astype(int).tolist()),
            epochAsDate=True,
        )
        return sts

    def _extract_slot(self, surveillance_result, slot_name):
        return np.asarray(surveillance_result.slots[slot_name])


class DisProgBasedAlgorithm(STSBasedAlgorithm):
    """Base class for algorithms that operate on the disProg (disease progress) class."""

    def _prepare_r_instance(self, data: pd.DataFrame):
        sts = super()._prepare_r_instance(data)
        return surveillance.sts2disProg(sts)

    def _extract_slot(self, surveillance_result, slot_name):
        names = surveillance_result.names
        names = names() if callable(names) else names
        return np.asarray(
            dict(zip(names, list(surveillance_result)))[slot_name]
        )
