from typing import List, Tuple
from pathlib import Path
from xml.etree import ElementTree
import json
import re

from bs4 import BeautifulSoup
import nltk


ANNOTATIONS_FOLDER = 'ere'
DOCUMENTS_FOLDER = 'source'


class Parser:
    def __init__(self, document_path: Path, annotation_paths: List[Path]):
        self.path = document_path
        self.entity_mentions = []
        self.event_mentions = []
        self.sentences = []
        self.document_text = ''

        self.entity_mentions, self.event_mentions = [], []
        # Sort the paths to make the order of the mentions deterministic
        for annotation_path in self._sorted_annotation_paths(document_path.stem, annotation_paths):
            more_entity_mentions, more_event_mentions = self.parse_xml(annotation_path)
            self.entity_mentions.extend(more_entity_mentions)
            self.event_mentions.extend(more_event_mentions)
        self.sents_with_pos = self.parse_document(document_path)
        self.fix_wrong_position()

    @staticmethod
    def _sorted_annotation_paths(document_name: str, annotation_paths: List[Path]) -> List[Path]:
        # We sort in increasing order by the start of the annotation range,
        # which is indicated in the file name by the first number after the document name.
        # The annotation file names look like
        # ${document_name}_${start}-${end}.richere.xml
        def sort_key(path: Path) -> int:
            # Use len + 1 to get rid of the underscore
            after_document_prefix = len(document_name) + 1
            before_end_of_number = path.name[after_document_prefix:].index('-')
            # before_end_of_number is relative to the sliced string so we have to treat it as an
            # offset
            return int(path.name[after_document_prefix:after_document_prefix + before_end_of_number])

        return sorted(annotation_paths, key=sort_key)

    @staticmethod
    def _get_document_filepath(richere_path: Path, document_name: str) -> Path:
        return (richere_path / DOCUMENTS_FOLDER / document_name).with_suffix('.xml')

    @staticmethod
    def _get_annotation_filepaths(richere_path: Path, document_name: str) -> List[Path]:
        return list((richere_path / ANNOTATIONS_FOLDER).glob(document_name + '*.richere.xml'))

    @staticmethod
    def from_data_path_and_name(richere_path: Path, document_name: str) -> "Parser":
        document_path, annotation_paths = Parser._get_and_verify_paths(richere_path, document_name)
        return Parser(document_path, annotation_paths)

    @staticmethod
    def _get_and_verify_paths(richere_path: Path, document_name: str) -> Tuple[Path, List[Path]]:
        document_path = Parser._get_document_filepath(richere_path, document_name)
        if not document_path.exists():
            raise RuntimeError(f"No annotations found in {document_path} for document {document_name}.")

        annotation_paths = Parser._get_annotation_filepaths(richere_path, document_name)
        if not annotation_paths:
            raise RuntimeError(f"No annotations found in {document_path} for document {document_name}.")
        return document_path, annotation_paths

    @staticmethod
    def verify_document(richere_path: Path, document_name: str):
        Parser._get_and_verify_paths(richere_path, document_name)

    @staticmethod
    def clean_text(text):
        return text.replace('\n', ' ')

    def get_data(self):
        data = []
        for sent in self.sents_with_pos:
            item = dict()

            item['sentence'] = self.clean_text(sent['text'])
            item['position'] = sent['position']
            text_position = sent['position']

            for i, s in enumerate(item['sentence']):
                if s != ' ':
                    item['position'][0] += i
                    break

            item['sentence'] = item['sentence'].strip()

            entity_map = dict()
            item['golden-entity-mentions'] = []
            item['golden-event-mentions'] = []

            for entity_mention in self.entity_mentions:
                entity_position = entity_mention['position']

                if text_position[0] <= entity_position[0] and entity_position[1] <= text_position[1]:

                    item['golden-entity-mentions'].append({
                        'text': self.clean_text(entity_mention['text']),
                        'position': entity_position,
                        'entity-type': entity_mention['entity-type'],
                        'head': {
                            "text": self.clean_text(entity_mention['head']["text"]),
                            "position": entity_mention["head"]["position"]
                        },
                        "entity_id": entity_mention['entity-id']
                    })
                    entity_map[entity_mention['entity-id']] = entity_mention

            for event_mention in self.event_mentions:
                event_position = event_mention['trigger']['position']
                if text_position[0] <= event_position[0] and event_position[1] <= text_position[1]:
                    event_arguments = []
                    for argument in event_mention['arguments']:
                        try:
                            entity_type = entity_map[argument['entity-id']]['entity-type']
                        except KeyError:
                            print('[Warning] The entity in the other sentence is mentioned. This argument will be ignored.')
                            continue

                        event_arguments.append({
                            'role': argument['role'],
                            'position': argument['position'],
                            'entity-type': entity_type,
                            'text': self.clean_text(argument['text']),
                        })

                    item['golden-event-mentions'].append({
                        'trigger': event_mention['trigger'],
                        'arguments': event_arguments,
                        'position': event_position,
                        'event_type': event_mention['event_type'],
                    })
            data.append(item)
        return data

    def find_correct_offset(self, sgm_text, start_index, text):
        offset = 0
        for i in range(0, 70):
            for j in [-1, 1]:
                offset = i * j
                if sgm_text[start_index + offset:start_index + offset + len(text)] == text:
                    return offset

        print('[Warning] fail to find offset! (start_index: {}, text: {}, path: {})'.format(start_index, text, self.path))
        return offset

    def fix_wrong_position(self):
        for entity_mention in self.entity_mentions:
            offset = self.find_correct_offset(
                sgm_text=self.document_text,
                start_index=entity_mention['position'][0],
                text=entity_mention['text'])

            entity_mention['position'][0] += offset
            entity_mention['position'][1] += offset
            entity_mention['head']["position"][0] += offset
            entity_mention['head']["position"][1] += offset

        for event_mention in self.event_mentions:
            offset1 = self.find_correct_offset(
                sgm_text=self.document_text,
                start_index=event_mention['trigger']['position'][0],
                text=event_mention['trigger']['text'])
            event_mention['trigger']['position'][0] += offset1
            event_mention['trigger']['position'][1] += offset1

            for argument in event_mention['arguments']:
                offset2 = self.find_correct_offset(
                    sgm_text=self.document_text,
                    start_index=argument['position'][0],
                    text=argument['text'])
                argument['position'][0] += offset2
                argument['position'][1] += offset2

    def parse_document(self, sgm_path):
        with open(sgm_path, 'r') as f:
            soup = BeautifulSoup(f.read(), features='html.parser')
            self.document_text = soup.text

            doc_type = soup.doc.doctype.text.strip()

            def remove_tags(selector):
                tags = soup.findAll(selector)
                for tag in tags:
                    tag.extract()

            if doc_type == 'WEB TEXT':
                remove_tags('poster')
                remove_tags('postdate')
                remove_tags('subject')
            elif doc_type in ['CONVERSATION', 'STORY']:
                remove_tags('speaker')

            sents = []
            converted_text = soup.text

            for sent in nltk.sent_tokenize(converted_text):
                sents.extend(sent.split('\n\n'))
            sents = list(filter(lambda x: len(x) > 5, sents))
            sents = sents[1:]
            sents_with_pos = []
            last_pos = 0
            for sent in sents:
                pos = self.document_text.find(sent, last_pos)
                last_pos = pos
                sents_with_pos.append({
                    'text': sent,
                    'position': [pos, pos + len(sent)]
                })

            return sents_with_pos

    def parse_xml(self, xml_path):
        entity_mentions, event_mentions = [], []
        tree = ElementTree.parse(xml_path)
        root = tree.getroot()

        for child in root[0]:
            if child.tag == 'entity':
                entity_mentions.extend(self.parse_entity_tag(child))
            elif child.tag in ['value', 'timex2']:
                entity_mentions.extend(self.parse_value_timex_tag(child))
            elif child.tag == 'event':
                event_mentions.extend(self.parse_event_tag(child))

        return entity_mentions, event_mentions

    @staticmethod
    def parse_entity_tag(node):
        entity_mentions = []

        for child in node:
            if child.tag != 'entity_mention':
                continue
            extent = child[0]
            head = child[1]
            charset = extent[0]
            head_charset = head[0]

            entity_mention = dict()
            entity_mention['entity-id'] = child.attrib['ID']
            entity_mention['entity-type'] = '{}:{}'.format(node.attrib['TYPE'], node.attrib['SUBTYPE'])
            entity_mention['text'] = charset.text
            entity_mention['position'] = [int(charset.attrib['START']), int(charset.attrib['END'])]
            entity_mention["head"] = {"text": head_charset.text,
                                      "position": [int(head_charset.attrib['START']), int(head_charset.attrib['END'])]}

            entity_mentions.append(entity_mention)

        return entity_mentions

    @staticmethod
    def parse_event_tag(node):
        event_mentions = []
        for child in node:
            if child.tag == 'event_mention':
                event_mention = dict()
                event_mention['event_type'] = '{}:{}'.format(node.attrib['TYPE'], node.attrib['SUBTYPE'])
                event_mention['arguments'] = []
                for child2 in child:
                    if child2.tag == 'ldc_scope':
                        charset = child2[0]
                        event_mention['text'] = charset.text
                        event_mention['position'] = [int(charset.attrib['START']), int(charset.attrib['END'])]
                    if child2.tag == 'anchor':
                        charset = child2[0]
                        event_mention['trigger'] = {
                            'text': charset.text,
                            'position': [int(charset.attrib['START']), int(charset.attrib['END'])],
                        }
                    if child2.tag == 'event_mention_argument':
                        extent = child2[0]
                        charset = extent[0]
                        event_mention['arguments'].append({
                            'text': charset.text,
                            'position': [int(charset.attrib['START']), int(charset.attrib['END'])],
                            'role': child2.attrib['ROLE'],
                            'entity-id': child2.attrib['REFID'],
                        })
                event_mentions.append(event_mention)
        return event_mentions

    @staticmethod
    def parse_value_timex_tag(node):
        entity_mentions = []

        for child in node:
            extent = child[0]
            charset = extent[0]

            entity_mention = dict()
            entity_mention['entity-id'] = child.attrib['ID']

            if 'TYPE' in node.attrib:
                entity_mention['entity-type'] = node.attrib['TYPE']
            if 'SUBTYPE' in node.attrib:
                entity_mention['entity-type'] += ':{}'.format(node.attrib['SUBTYPE'])
            if child.tag == 'timex2_mention':
                entity_mention['entity-type'] = 'TIM:time'

            entity_mention['text'] = charset.text
            entity_mention['position'] = [int(charset.attrib['START']), int(charset.attrib['END'])]

            entity_mention["head"] = {"text": charset.text,
                                      "position": [int(charset.attrib['START']), int(charset.attrib['END'])]}

            entity_mentions.append(entity_mention)

        return entity_mentions


if __name__ == '__main__':
    # parser = Parser('./data/ace_2005_td_v7/data/English/un/fp2/alt.gossip.celebrities_20041118.2331')
    parser = Parser('./data/ace_2005_td_v7/data/English/un/timex2norm/alt.corel_20041228.0503')
    data = parser.get_data()
    with open('./output/debug.json', 'w') as f:
        json.dump(data, f, indent=2)

    # index = parser.sgm_text.find("Diego Garcia")
    # print('index :', index)
    # print(parser.sgm_text[1918 - 30:])