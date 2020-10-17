# -*- coding: utf-8 -*-
#
# This file is part of RERO MEF.
# Copyright (C) 2017 RERO.
#
# RERO MEF is free software; you can redistribute it
# and/or modify it under the terms of the GNU General Public License as
# published by the Free Software Foundation; either version 2 of the
# License, or (at your option) any later version.
#
# RERO MEF is distributed in the hope that it will be
# useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with RERO MEF; if not, write to the
# Free Software Foundation, Inc., 59 Temple Place, Suite 330, Boston,
# MA 02111-1307, USA.
#
# In applying this license, RERO does not
# waive the privileges and immunities granted to it by virtue of its status
# as an Intergovernmental Organization or submit itself to any jurisdiction.

"""RERO MEF invenio module declaration."""

from __future__ import absolute_import, print_function

from invenio_indexer.signals import before_record_index

from .mef.listner import enrich_mef_data


class REROMEFAPP(object):
    """rero-mef extension."""

    def __init__(self, app=None):
        """RERO MEF App module."""
        if app:
            self.init_app(app)
            self.register_signals(app)

    def init_app(self, app):
        """Flask application initialization."""
        app.extensions['rero-mef'] = self

    def register_signals(self, app):
        """Register signals."""
        before_record_index.connect(enrich_mef_data, sender=app)
