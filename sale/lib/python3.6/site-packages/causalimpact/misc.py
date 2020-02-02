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

"""Miscellaneous functions to help in the implementation of Causal Impact."""


from __future__ import absolute_import, division, print_function

import scipy.stats as stats
from statsmodels.tsa.statespace.structural import UnobservedComponents


def standardize(data):
    """
    Applies standardization to input data. Result should have mean zero and standard
    deviation of one.

    Args
    ----
      data: pandas DataFrame.

    Returns
    -------
      list:
        data: standardized data with zero mean and std of one.
        tuple:
          mean and standard deviation used on each column of input data to make
          standardization. These values should be used to obtain the original dataframe.

    Raises
    ------
      ValueError: if data has only one value.
    """
    if data.shape[0] == 1:
        raise ValueError('Input data must have more than one value')
    mu = data.mean(skipna=True)
    std = data.std(skipna=True, ddof=0)
    data = (data - mu) / std.fillna(1)
    return [data, (mu, std)]


def unstandardize(data, mus_sigs):
    """
    Applies the inverse transformation to return to original data.

    Args
    ----
      data: pandas DataFrame with zero mean and std of one.
      mus_sigs: tuple where first value is the mean used for the standardization and
                second value is the respective standard deviaion.

    Returns
    -------
      data: pandas DataFrame with mean and std given by input ``mus_sigs``
    """
    mu, sig = mus_sigs
    data = (data * sig) + mu
    return data


def get_z_score(p):
    """
    Returns the correspondent z-score with probability area p.

    Args
    ----
      p: float ranging between 0 and 1 representing the probability area to convert.

    Returns
    -------
      The z-score correspondent of p.
    """
    return stats.norm.ppf(p)


def get_reference_model(model, endog, exog):
    """
    Build an `UnobservedComponents` model using as reference the input `model`. We need
    an exactly similar object as `model` but instantiated with different `endog` and
    `exog`.

    Args
    ----
      model: `UnobservedComponents`.
          Template model that is used as reference to build a new one with new `endog`
          and `exog` variables.
      endog: pandas.Series.
          New endog value to be used in model.
      exog: pandas.Series.
          New exog value to be used in model. If original model does not contain
          exogenous variables then it's not set in `ref_model`.

    Returns
    -------
      ref_model: `UnobservedComponents`.
          New model built from input `model` setup.
    """
    model_args = model._get_init_kwds()
    model_args['endog'] = endog
    if model.exog is not None:
        model_args['exog'] = exog
    ref_model = UnobservedComponents(**model_args)
    return ref_model
