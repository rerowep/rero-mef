# -*- coding: utf-8 -*-
#
# RERO MEF
# Copyright (C) 2021 RERO
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, version 3 of the License.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

"""API for manipulating MEF records."""

from copy import deepcopy

from flask import current_app
from invenio_search.api import RecordsSearch

from .fetchers import mef_id_fetcher
from .minters import mef_id_minter
from .models import ConceptMefMetadata
from .providers import ConceptMefProvider
from ...api import ReroIndexer
from ...api_mef import EntityMefRecord


def build_ref_string(concept_pid, concept):
    """Build url for concept's api.

    :param concept_pid: Pid of concept.
    :param concept: Type of concept.
    :returns: Reference string to record.
    """
    with current_app.app_context():
        return (f'{current_app.config.get("RERO_MEF_APP_BASE_URL")}'
                f'/api/concepts/{concept}/{concept_pid}')


class ConceptMefSearch(RecordsSearch):
    """RecordsSearch."""

    class Meta:
        """Search only on index."""

        index = 'concepts_mef'
        doc_types = None
        fields = ('*', )
        facets = {}

        default_filter = None


class ConceptMefRecord(EntityMefRecord):
    """Mef concept class."""

    minter = mef_id_minter
    fetcher = mef_id_fetcher
    provider = ConceptMefProvider
    name = 'mef'
    model_cls = ConceptMefMetadata
    search = ConceptMefSearch
    mef_type = 'CONCEPTS'
    entities = ['idref', 'rero']

    def replace_refs(self):
        """Replace $ref with real data."""
        data = deepcopy(self)
        data = super().replace_refs()
        sources = []
        for concept in self.entities:
            if data.get(concept):
                sources.append(concept)
                if metadata := data[concept].get('metadata'):
                    data[concept] = metadata
        data['sources'] = sources
        return data

    @classmethod
    def get_latest(cls, pid_type, pid):
        """Get latest Mef record for pid_type and pid.

        :param pid_type: pid type to use.
        :param pid: pid to use..
        :returns: latest record.
        """
        search = ConceptMefSearch().filter({'term': {f'{pid_type}.pid': pid}})
        if search.count() > 0:
            data = next(search.scan()).to_dict()
            new_pid = None
            if relation_pid := data.get(pid_type, {}).get('relation_pid'):
                if relation_pid['type'] == 'redirect_to':
                    new_pid = relation_pid['value']
            elif pid_type == 'idref':
                # Find new pid from redirect_pid redirect_from
                search = ConceptMefSearch() \
                        .filter('term', idref__relation_pid__value=pid)
                if search.count() > 0:
                    new_data = next(search.scan()).to_dict()
                    new_pid = new_data.get('idref', {}).get('pid')
            return cls.get_latest(pid_type=pid_type, pid=new_pid) \
                if new_pid else data
        return {}


class ConceptMefIndexer(ReroIndexer):
    """MefIndexer."""

    record_class = ConceptMefRecord

    def bulk_index(self, record_id_iterator):
        """Bulk index records.

        :param record_id_iterator: Iterator yielding record UUIDs.
        """
        self._bulk_op(record_id_iterator, op_type='index', doc_type='mef')
