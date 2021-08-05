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

"""Record serialization."""

from flask import request, url_for
from invenio_records_rest.links import default_links_factory_with_additional
from invenio_records_rest.schemas import RecordSchemaJSONV1
from invenio_records_rest.serializers.json import JSONSerializer
from invenio_records_rest.serializers.response import record_responsify

from ..mef.api import AgentMefSearch


def add_links(pid, record):
    """Add MEF link to VIAF."""
    links = {}
    viaf_pid = record.get('pid')
    mef_pid_search = AgentMefSearch() \
        .filter('term', viaf_pid=viaf_pid) \
        .source(['pid']).scan()
    try:
        for idx, search in enumerate(mef_pid_search):
            url = '{scheme}://{host}/api/agents/mef/' + str(search.pid)
            if idx:
                links[f'mef {idx}'] = url
            else:
                links['mef'] = url
    except Exception:
        pass
    links['viaf.org'] = 'http://www.viaf.org/viaf/' + str(viaf_pid)

    link_factory = default_links_factory_with_additional(links)
    return link_factory(pid)


# Nice to have direct working links in test server!
def local_link(agent, name, record):
    """Change links to actual links."""
    if name in record:
        ref = record[name].get('$ref')
        if ref:
            my_pid = ref.split('/')[-1]
            url = url_for(
                f'invenio_records_rest.{agent}_item',
                pid_value=my_pid,
                _external=True
            )
            record[name].update({'$ref': url})


class ReroMefSerializer(JSONSerializer):
    """Mixin serializing records as JSON."""

    def serialize(self, pid, record, links_factory=None, **kwargs):
        """Serialize a single record and persistent identifier.

        :param pid: Persistent identifier instance.
        :param record: Record instance.
        :param links_factory: Factory function for record links.
        """
        if request and request.args.get('resolve'):
            record = record.replace_refs()

        return super(ReroMefSerializer, self).serialize(
            pid, record, links_factory=add_links, **kwargs
        )


json_v1 = ReroMefSerializer(RecordSchemaJSONV1)
"""JSON v1 serializer."""

json_v1_agent_viaf_response = record_responsify(
    json_v1, 'application/rero+json')
