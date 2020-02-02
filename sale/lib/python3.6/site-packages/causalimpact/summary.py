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
Summarizes performance information inferred in post-inferences compilation process.
"""


from __future__ import absolute_import, division, print_function

import os

from jinja2 import Template

from causalimpact.misc import get_z_score

_here = os.path.dirname(os.path.abspath(__file__))
summary_tmpl_path = os.path.join(_here, 'templates', 'summary')
report_tmpl_path = os.path.join(_here, 'templates', 'report')

SUMMARY_TMPL = Template(open(summary_tmpl_path).read())
REPORT_TMPL = Template(open(report_tmpl_path).read())


class Summary(object):
    """
    Prepares final summary with causal impact results telling whether an effect has been
    identified in data or not.
    """
    def __init__(self):
        self.summary_data = None

    def summary(self, output='summary', digits=2):
        """
        Returns final results from causal impact analysis, such as absolute observed
        effect, the relative effect between prediction and observed variable, cumulative
        performances in post-intervention period among other metrics.

        Args
        ----
          output: str.
              Can be either "summary" or "report". The first is a simpler output just
              informing general metrics such as expected absolute or relative effect.

          digits: int.
              Defines the number of digits after the decimal point to round. For
              digits=2, value 1.566 becomes 1.57.

        Returns
        -------
          summary: str.
              Contains results of the causal impact analysis.

        Raises
        ------
          RuntimeError: if `self.summary_data` is None meaning the post inference
              compilation was not performed yet.
        """
        if self.summary_data is None:
            raise RuntimeError('Posterior inferences must be first computed before '
                               'running summary.')
        if output not in {'summary', 'report'}:
            raise ValueError('Please choose either summary or report for output.')
        if output == 'summary':
            summary = SUMMARY_TMPL.render(
                summary=self.summary_data.to_dict(),
                alpha=self.alpha,
                z_score=get_z_score(1 - self.alpha / 2.),
                p_value=self.p_value,
                digits=digits
            )
        else:
            summary = REPORT_TMPL.render(
                summary=self.summary_data.to_dict(),
                alpha=self.alpha,
                p_value=self.p_value,
                digits=digits
            )
        return summary
