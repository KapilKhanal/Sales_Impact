# Copyright 2014 Google Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Causal Impact class for running impact inferences caused in a time evolving system.
"""


from __future__ import absolute_import, division, print_function

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.structural import UnobservedComponents

from causalimpact.inferences import Inferences
from causalimpact.misc import standardize
from causalimpact.plot import Plot
from causalimpact.summary import Summary


class BaseCausal(Inferences, Summary, Plot):
    """
    Works as a container for attributes and methods that are used in the Causal
    Impact algorithm. Offers support for inferences, summary report and plotting
    functionality.

    Args
    ----
      data: pandas DataFrame.
          Input data processed and confirmed to be appropriate to be used in the Causal
          Impact algorithm.
      pre_period: list.
          Containing validated pre-intervention intervals.
      post_period: list.
          Containing validated post-intervention intervals.
      pre_data: pandas DataFrame.
          Sliced data regarding the pre-intervention period.
      post_data: pandas DataFrame.
          Sliced data regarding post-intervention period.
      alpha: float.
          Indicating significance level for hypothesis testing.
      mu_sig: list.
          With two values where first is the mean used to normalize `pre_data` and
          second value is the standard deviation also used in the normalization.
    """
    def __init__(self, data, pre_period, post_period, pre_data, post_data, alpha,
                 **kwargs):
        Inferences.__init__(self, n_sims=kwargs.get('n_sims', 1000))
        Summary.__init__(self)
        self.data = data
        self.pre_period = pre_period
        self.post_period = post_period
        self.pre_data = pre_data
        self.post_data = post_data
        self.alpha = alpha
        self.normed_pre_data = None
        self.normed_post_data = None
        self.mu_sig = None


class CausalImpact(BaseCausal):
    """
    Main class used to run the Causal Impact algorithm implemented by Google as
    described in the paper:

    https://google.github.io/CausalImpact/CausalImpact.html

    The main difference between Google's R package and Python's is that in the latter the
    optimization will be performed by using Kalman Filters as implemented in `statsmodels`
    package, contrary to the Markov Chain Monte Carlo technique used in R.

    Despite the different techniques, results should converge to the same optimum state
    space.

    Args
    ----
      data: numpy array, pandas DataFrame.
          First column must contain the `y` measured value while the others contain the
          covariates `X` that are used in the linear regression component of the model.
          If it's a pandas DataFrame, its index can be defined either as a `RangeIndex`,
          an `Index` or `DateTimeIndex`.
          In case of the second, then a conversion to `DateTime` type is automatically
          performed; in case of failure, the original index is kept as is.
      pre_period: list.
          A list of size two containing either `int`, `str` or `pd.Timestamp`  values
          that references the first time point in the trained data up to the last one
          to be used in the pre-intervention period for training the model.
          For example, valid inputs are:
            - [0, 30]
            - ['20180901', '20180930']
            - [pd.to_datetime('20180901'), pd.to_datetime('20180930')]
            - [pd.Timestamp('20180901'), pd.Timestamp('20180930')]
          where `pd` is the pandas module.
          The latter can be used only if the input `data` is a pandas DataFrame whose
          index is time based.
          Ideally, it should slice the data up to when the intervention started so that
          the trained model can more precisely predict what should have happened in the
          post-intervention period had no interference taken place.
      post_period: list.
          The same as `pre_period` but references where the post-intervention
          data begins and ends. This is the part of `data` used to make inferences.
      model: `statsmodels.tsa.statespace.structural.UnobservedComponents`.
          If a customized model is desired than this argument can be used
          otherwise a default 'local level' model is internally built. When using a user-
          defined model, it's still required to send `data` as input even though the
          pre-intervention period is already present in the model `endog` and `exog`
          attributes. We do so to keep the contract of the method simpler.
      alpha: float.
          A float that ranges between 0 and 1 indicating the significance level that
          will be used when statistically testing for signal presencen in the post-
          intervention period.
      kwargs:
        standardize: bool.
            If `True`, applies standardizes data to have zero mean and unitary standard
            deviation.
        disp: bool.
            Whether to print log associated to the `fit` method or not. `False` means no
            printing.
        prior_level_sd: float.
            Prior value for the local level standard deviation. If the explicit value of
            `None` is sent then an automatic optimization of the local level will take
            place. This is recommended when there's uncertainty about what prior value is
            appropriate for the data. In general, if the exogenous values are good
            descriptors of the observed response then this value can be low
            (such as the default of 0.01). In cases where there's not a complete
            correlation between exogenous and endogenous variables, the value 0.1 can be
            used, as suggested by Google. If no value is chosen at all, the value of
            `0.01` will be used as default value.
        nseasons: list of dicts.
            Models for `n` seasonal components in input response data. A seasonal
            component can be described as a pattern that repeats itself with peridiocity
            `s`. In `statsmodels` library, we have the option of doing so by using either
            the parameter `seasonal`, which uses `(s-1)` variables for each point of the
            series, or `freq_seasonal`, which is the one used in this package.
            The difference is that in the latter the equations are expressed in the
            frequency domain and accepts more than one seasonal component, such as a
            weekly and another monthly ones. If, for instance, in the input daily data has
            a known weekly and a montly seasonal components, then this paramter can be
            used like:
            `nseasons=[{'period': 7}, {'period': 30}]`. You can also specify how many
            harmonics should be used to express the final value, such as:
            `nseasons=[{'period': 7, 'harmonics': 3}, {'period': 30, 'harmonics': 5}]`.
            If no value is used for `harmonics`, its total amount `h` will be considered
            to be :math:`floor(s/2)`. Default value is [] meaning no seasonal component
            should be modeled in the fitting process. For more information, please refer
            to statsmodels docs:

            https://www.statsmodels.org/dev/generated/statsmodels.tsa.statespace.structural.UnobservedComponents.html
            If a custom model is used then it should already contain the definition of
            the seasonal components.

    Returns
    -------
      CausalImpact object with infereces already processed.

    Examples:
    ---------
      >>> import numpy as np
      >>> from statsmodels.tsa.statespace.structural import UnobservedComponents
      >>> from statsmodels.tsa.arima_process import ArmaProcess

      >>> np.random.seed(12345)
      >>> ar = np.r_[1, 0.9]
      >>> ma = np.array([1])
      >>> arma_process = ArmaProcess(ar, ma)
      >>> X = 100 + arma_process.generate_sample(nsample=100)
      >>> y = 1.2 * X + np.random.normal(size=100)
      >>> data = pd.DataFrame({'y': y, 'X': X}, columns=['y', 'X'])
      >>> pre_period = [0, 69]
      >>> post_period = [70, 99]

      >>> ci = CausalImpact(data, pre_period, post_period)
      >>> ci.summary()
      >>> ci.summary('report')
      >>> ci.plot()

      Using pandas DataFrames:

      >>> df = pd.DataFrame(data)
      >>> df = df.set_index(pd.date_range(start='20180101', periods=len(data)))
      >>> pre_period = ['20180101', '20180311']
      >>> post_period = ['20180312', '20180410']
      >>> ci = CausalImpact(df, pre_period, post_period)

      Using pandas DataFrames with pandas timestamps:

      >>> df = pd.DataFrame(data)
      >>> df = df.set_index(pd.date_range(start='20180101', periods=len(data)))
      >>> pre_period = [pd.to_datetime('20180101'), pd.to_datetime('20180311')]
      >>> post_period = [pd.to_datetime('20180312'), pd.to_datetime('20180410')]
      >>> ci = CausalImpact(df, pre_period, post_period)

      Using automatic local level optimization:

      >>> df = pd.DataFrame(data)
      >>> df = df.set_index(pd.date_range(start='20180101', periods=len(data)))
      >>> pre_period = ['20180101', '20180311']
      >>> post_period = ['20180312', '20180410']
      >>> ci = CausalImpact(df, pre_period, post_period, prior_level_sd=None)

      Using seasonal components:

      >>> df = pd.DataFrame(data)
      >>> df = df.set_index(pd.date_range(start='20180101', periods=len(data)))
      >>> pre_period = ['20180101', '20180311']
      >>> post_period = ['20180312', '20180410']
      >>> ci = CausalImpact(df, pre_period, post_period, nseasons=[{'period': 7}])

      Using a customized model:

      >>> pre_y = data[:70, 0]
      >>> pre_X = data[:70, 1:]
      >>> ucm = UnobservedComponents(endog=pre_y, level='llevel', exog=pre_X)
      >>> ci = CausalImpact(data, pre_period, post_period, model=ucm)
    """
    def __init__(self, data, pre_period, post_period, model=None, alpha=0.05, **kwargs):
        checked_input = self._process_input_data(
            data, pre_period, post_period, model, alpha, **kwargs
        )
        super(CausalImpact, self).__init__(**checked_input)
        self.model_args = checked_input['model_args']
        self.model = checked_input['model']
        self._fit_model()
        self._process_posterior_inferences()

    @property
    def model_args(self):
        """
        Gets the general settings used to guide the creation of the Causal model.

        Returns
        -------
          dict:
            standardize: bool.
        """
        return self._model_args

    @model_args.setter
    def model_args(self, value):
        """
        Sets general settings for how to build the Causal model.

        Args
        ----
          value: dict
              standardize: bool.
              nseasons: list of dicts.
        """
        if value.get('standardize'):
            self._standardize_pre_post_data()
        self._model_args = value

    @property
    def model(self):
        """
        Gets UnobservedComponents model that will be used for computing the Causal
        Impact algorithm.
        """
        return self._model

    @model.setter
    def model(self, value):
        """
        Sets model object.

        Args
        ----
          value: `UnobservedComponents`.
        """
        if value is None:
            self._model = self._get_default_model()
        else:
            self._model = value

    def _fit_model(self):
        """
        Uses the built model, prepares the arguments and fits the kalman filter for the
        inferences phase.
        """
        fit_args = self._process_fit_args()
        self.trained_model = self.model.fit(**fit_args)

    def _standardize_pre_post_data(self):
        """
        Applies normal standardization in pre and post data, based on mean and std of
        pre-data (as it's used for training our model). Sets new values for
        `self.pre_data`, `self.post_data`, `self.mu_sig`.
        """
        self.normed_pre_data, (mu, sig) = standardize(self.pre_data)
        self.normed_post_data = (self.post_data - mu) / sig
        self.mu_sig = (mu[0], sig[0])

    def _process_posterior_inferences(self):
        """
        Uses the trained model to make predictions for the post-intervention (or test
        data) period by invoking the class `Inferences` to process the forecasts. All
        data related to predictions, point effects and cumulative responses will be
        processed here.
        """
        self._compile_posterior_inferences()
        self._summarize_posterior_inferences()

    def _get_default_model(self):
        """Constructs default local level unobserved states model using input data and
        `self.model_args`.

        Returns
        -------
          model: `UnobservedComponents` built using pre-intervention data as training
              data.
        """
        data = self.pre_data if self.normed_pre_data is None else self.normed_pre_data
        y = data.iloc[:, 0]
        X = data.iloc[:, 1:] if data.shape[1] > 1 else None
        freq_seasonal = self.model_args.get('nseasons')
        model = UnobservedComponents(endog=y, level='llevel', exog=X,
                                     freq_seasonal=freq_seasonal)
        return model

    def _process_input_data(self, data, pre_period, post_period, model, alpha, **kwargs):
        """
        Checks and formats when appropriate the input data for running the Causal
        Impact algorithm. Performs assertions such as missing or invalid arguments.

        Args
        ----
          data: numpy.array, pandas.DataFrame.
              First column is the response variable `y` and other columns correspond to
              the covariates `X`.
          pre_data: numpy.array, pandas.DataFrame.
              Pre-intervention data sliced from input data.
          post_data: numpy.array, pandas.DataFrame.
              Post_intervention data sliced from input data.
          model: None, UnobservedComponents.
          alpha: float.
          kwargs:
            standardize: bool.
            disp: bool.
            prior_level_sd: float.
            nseasons: list of dicts.

        Returns
        -------
          dict of:
            data: pandas DataFrame.
                Validated data, first column is `y` and the others is the `X` covariates.
            pre_data: pandas DataFrame.
                Data sliced using `pre_period` values.
            post_data: pandas DataFrame.
            model: Either `None` or `UnobservedComponents` validated to be correct.
            alpha: float ranging from 0 to 1.
            model_args: dict containing general information related to how to process
                the causal impact algorithm.

        Raises
        ------
          ValueError: if input arguments is `None`.
        """
        input_args = locals().copy()
        model = input_args.pop('model')
        none_args = [arg for arg, value in input_args.items() if value is None]
        if none_args:
            raise ValueError('{args} input cannot be empty'.format(
                             args=', '.join(none_args)))
        processed_data = self._format_input_data(data)
        pre_data, post_data = self._process_pre_post_data(processed_data, pre_period,
                                                          post_period)
        alpha = self._process_alpha(alpha)
        model_args = self._process_model_args(**kwargs)
        if model:
            model = self._process_input_model(model)
        return {
            'data': processed_data,
            'pre_period': pre_period,
            'post_period': post_period,
            'pre_data': pre_data,
            'post_data': post_data,
            'model': model,
            'alpha': alpha,
            'model_args':  model_args
        }

    def _process_fit_args(self):
        """
        Process the input that will be used in the fitting process for the model.

        Args
        ----
          self:
            model: `UnobservedComponents` from statsmodels.
                If `None` them it means the fitting process will work with default model.
                Process level information of customized model otherwise.
            model_args: dict.
                Input args for general options of the model. All keywords defined
                in `scipy.optimize.minimize` can be used here. For more details,
                please refer to:
                https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html

              disp: bool.
                  Whether to display the logging of the `statsmodels` fitting process or
                  not. Defaults to `False` which means not display any logging.

              prior_level_sd: float.
                  Prior value to be used as reference for the fitting process.

        Returns
        -------
          model_args: dict
              The arguments that will be used in the `fit` method.
        """
        fit_args = self.model_args.copy()
        fit_args.setdefault('disp', False)
        level_sd = fit_args.get('prior_level_sd', 0.01)
        n_params = len(self.model.param_names)
        level_idx = [idx for (idx, name) in enumerate(self.model.param_names) if
                     name == 'sigma2.level']
        bounds = [(None, None)] * n_params
        if level_idx:  # If chosen model do not have level defined then this is None.
            level_idx = level_idx[0]
            # We make the maximum relative variation be up to 20% in order to simulate
            # an approximate behavior of the respective algorithm implemented in R.
            bounds[level_idx] = (
                level_sd / 1.2 if level_sd is not None else None,
                level_sd * 1.2 if level_sd is not None else None
            )
        fit_args.setdefault('bounds', bounds)
        return fit_args

    def _validate_y(self, y):
        """
        Validates if input response variable is correct and doesn't have invalid input.

        Args
        ----
          y: pandas Series.
             Response variable sent in input data in first column.

        Raises
        ------
          ValueError: if values in `y` are Null.
                      if less than 3 (three) non-null values in `y` (as in this case
                          we can't even train a model).
                      if `y` is constant (in this case it doesn't make much sense to
                        make predictions as the time series doesn't change in the training
                        phase.
        """
        if np.all(y.isna()):
            raise ValueError('Input response cannot have just Null values.')
        if y.notna().values.sum() < 3:
            raise ValueError('Input response must have more than 3 non-null '
                             'points at least.')
        if y.std(skipna=True, ddof=0) == 0:
            raise ValueError('Input response cannot be constant.')

    def _process_alpha(self, alpha):
        """
        Asserts input `alpha` is appropriate to be used in the model.

        Args
        ----
          alpha: float.
              Ranges from 0 up to 1 indicating level of significance to assert when
              testing for presence of signal in post-intervention data.

        Returns
        -------
          alpha: float.
              Validated `alpha` value.

        Raises
        ------
          ValueError: if alpha is not float.
                      if alpha is not between 0. and 1.
        """
        if not isinstance(alpha, float):
            raise ValueError('alpha must be of type float.')
        if alpha < 0 or alpha > 1:
            raise ValueError(
                'alpha must range between 0 (zero) and 1 (one) inclusive.'
            )
        return alpha

    def _process_input_model(self, model):
        """
        Checkes whether input model was properly built and is ready to be run.

        Args
        ----
          model: `UnobservedComponents`.

        Returns
        -------
          model: `UnobservedComponents`.
              Validated model.

        Raises
        ------
          ValueError: if model is not of appropriate type.
                      if model doesn't have attribute level or it's not set.
                      if model doesn't have attribute exog or it's not set.
                      if model doesn't have attribute data or it's not set.
        """
        if not isinstance(model, UnobservedComponents):
            raise ValueError('Input model must be of type UnobservedComponents.')
        if not model.level:
            raise ValueError('Model must have level attribute set.')
        if model.exog is None:
            raise ValueError('Model must have exog attribute set.')
        if model.data is None:
            raise ValueError('Model must have data attribute set.')
        return model

    def _process_model_args(self, **kwargs):
        """
        Process general parameters related to how Causal Impact will be implemented, such
        as standardization procedure or the addition of seasonal components to the model.

        Args
        ----
          kwargs:
            standardize: bool.
            nseasons: list of dicts.
            other keys used in fitting process.

        Returns
        -------
          dict of:
            standardize: bool.
            nseasons: list of dicts.
            other keys used in fitting process.

        Raises
        ------
          ValueError: if standardize is not of type `bool`.
                      if nseasons doesn't follow the pattern [{str key: number}].
        """
        standardize = kwargs.get('standardize')
        if standardize is None:
            standardize = True  # Default behaviour is to set standardization to True.
        if not isinstance(standardize, bool):
            raise ValueError('Standardize argument must be of type bool.')
        kwargs['standardize'] = standardize
        nseasons = kwargs.get('nseasons')
        if nseasons is None:
            nseasons = []
        for season in nseasons:
            if not isinstance(season, dict):
                raise ValueError(
                    'nseasons must be a list of dicts with the required key "period" '
                    'and the optional key "harmonics".'
                )
            if 'period' not in season:
                raise ValueError('nseasons dicts must contain the key "period" defined.')
            if 'harmonics' in season:
                if season.get('harmonics') > season['period'] / 2:
                    raise ValueError(
                        'Total harmonics must be less or equal than periods '
                        'divided by 2.'
                    )
        kwargs['nseasons'] = nseasons
        return kwargs

    def _format_input_data(self, data):
        """
        Validates and formats input data.

        Args
        ----
          data: `numpy.array` or `pandas.DataFrame`.

        Returns
        -------
          data: pandas DataFrame.
              Validated data to be used in Causal Impact algorithm.

        Raises
        ------
          ValueError: if input `data` is non-convertible to pandas DataFrame.
                      if input `data` has non-numeric values.
                      if input `data` has less than 3 points.
                      if input covariates have NAN values.
        """
        if not isinstance(data, pd.DataFrame):
            try:
                data = pd.DataFrame(data)
            except ValueError:
                raise ValueError(
                    'Could not transform input data to pandas DataFrame.'
                )
        self._validate_y(data.iloc[:, 0])
        # Must contain only numeric values
        if not data.applymap(np.isreal).values.all():
            raise ValueError('Input data must contain only numeric values.')
        # Covariates cannot have NAN values
        if data.shape[1] > 1:
            if data.iloc[:, 1:].isna().values.any():
                raise ValueError('Input data cannot have NAN values.')
        # If index is a string of dates, try to convert it to datetimes which helps
        # in plotting.
        data = self._convert_index_to_datetime(data)
        return data

    def _convert_index_to_datetime(self, data):
        """
        If input data has index of string dates, i.e, '20180101', '20180102'..., try
        to convert it to datetime specifically, which results in
        Timestamp('2018-01-01 00:00:00'), Timestamp('2018-01-02 00:00:00')

        Args
        ----
          data: pandas DataFrame
              Input data used in causal impact analysis.

        Returns
        -------
          data: pandas DataFrame
              Same input data with potentially new index of type DateTime.
        """
        if isinstance(data.index.values[0], str):
            try:
                data.set_index(pd.to_datetime(data.index), inplace=True)
            except ValueError:
                pass
        return data

    def _process_pre_post_data(self, data, pre_period, post_period):
        """
        Checks `pre_period`, `post_period` and returns data sliced accordingly to  each
        period.

        Args
        ----
          data: pandas DataFrame.
          pre_period: list.
              Contains either `int` or `str` values.
          post_period: same as `pre_period`.

        Returns
        -------
          result: list.
              First value is pre-intervention data and second value is post-intervention.

        Raises
        ------
          ValueError: if pre_period last value is bigger than post intervention period.
        """
        checked_pre_period = self._process_period(pre_period, data)
        checked_post_period = self._process_period(post_period, data)

        if checked_pre_period[1] > checked_post_period[0]:
            raise ValueError(
                'Values in training data cannot be present in the post-intervention '
                'data. Please fix your pre_period value to cover at most one point less '
                'from when the intervention happened.'
            )
        if checked_pre_period[1] < checked_pre_period[0]:
            raise ValueError('pre_period last number must be bigger than its first.')
        if checked_pre_period[1] - checked_pre_period[0] < 3:
            raise ValueError('pre_period must span at least 3 time points.')
        if checked_post_period[1] < checked_post_period[0]:
            raise ValueError('post_period last number must be bigger than its first.')
        result = [
            data.loc[pre_period[0]: pre_period[1], :],
            data.loc[post_period[0]: post_period[1], :]
        ]
        return result

    def _process_period(self, period, data):
        """
        Validates period inputs.

        Args
        ----
          period: list.
              Containing two values that can be either `int`, `str` or `pd.Timestamp`
          data: pandas DataFrame.
              Input Causal Impact data.

        Returns
        -------
          period: list.
              Validated period list.

        Raises
        ------
          ValueError: if input `period` is not of type list.
                      if input doesn't have two elements.
                      if period date values are not present in data.
        """
        if not isinstance(period, list):
            raise ValueError('Input period must be of type list.')
        if len(period) != 2:
            raise ValueError(
                'Period must have two values regarding the beginning and end of '
                'the pre and post intervention data.'
            )
        none_args = [d for d in period if d is None]
        if none_args:
            raise ValueError('Input period cannot have `None` values.')
        if not (
            (isinstance(period[0], int) and isinstance(period[1], int)) or
            (isinstance(period[1], str) and isinstance(period[1], str)) or
            (isinstance(period[1], pd.Timestamp) and isinstance(period[1], pd.Timestamp))
        ):
            raise ValueError('Input must contain either int, str or pandas Timestamp')
        # Tests whether the input period is indeed present in the input data index.
        for point in period:
            if point not in data.index:
                if isinstance(point, pd.Timestamp):
                    point = point.strftime('%Y%m%d')
                raise ValueError("{point} not present in input data index.".format(
                    point=str(point)
                    )
                )
        if isinstance(period[0], str) or isinstance(period[0], pd.Timestamp):
            period = self._convert_str_period_to_int(period, data)
        return period

    def _convert_str_period_to_int(self, period, data):
        """
        Converts string values from `period` to integer offsets from `data`.

        Args
        ----
          period: list of str or pandas timestamps
          data: pandas DataFrame.

        Returns
        -------
          period: list of int.
              Where each value is the correspondent integer based value in `data` index.
        """
        result = []
        for date in period:
            offset = data.index.get_loc(date)
            result.append(offset)
        return result
