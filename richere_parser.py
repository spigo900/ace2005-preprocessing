import json
import logging
from pathlib import Path
from typing import List, Mapping, Tuple
from xml.etree import ElementTree

from bs4 import BeautifulSoup, NavigableString
import nltk

nltk.download('punkt')

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


ANNOTATIONS_DIRECTORY = 'ere'
DOCUMENTS_DIRECTORY = 'source'


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
            more_entity_mentions, more_event_mentions = self.parse_annotations(annotation_path)
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
        return (richere_path / DOCUMENTS_DIRECTORY / document_name).with_suffix('.xml')

    @staticmethod
    def _get_annotation_filepaths(richere_path: Path, document_name: str) -> List[Path]:
        return list((richere_path / ANNOTATIONS_DIRECTORY).glob(document_name + '*.rich_ere.xml'))

    @staticmethod
    def from_data_path_and_name(richere_path: Path, document_name: str) -> "Parser":
        document_path, annotation_paths = Parser._get_and_verify_paths(richere_path, document_name)
        return Parser(document_path, annotation_paths)

    @staticmethod
    def _get_and_verify_paths(richere_path: Path, document_name: str) -> Tuple[Path, List[Path]]:
        document_path = Parser._get_document_filepath(richere_path, document_name)
        if not document_path.exists():
            raise RuntimeError(f"No such document {document_name} found at {document_path} in {richere_path}.")

        annotation_paths = Parser._get_annotation_filepaths(richere_path, document_name)
        if not annotation_paths:
            raise RuntimeError(f"No annotations found in {richere_path} for document {document_name}.")
        return document_path, annotation_paths

    @staticmethod
    def verify_document(richere_path: Path, document_name: str):
        Parser._get_and_verify_paths(richere_path, document_name)

    @staticmethod
    def clean_text(text):
        return text.replace('\n', ' ')

    def get_data(self):
        """
        Return a list of (sentence, annotations) pairs for every sentence in the document.
        """
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

            # Maintain a mapping from entity IDs to entity mentions.
            #
            # Note that this only maps the entity ID to SOME mention in the sentence; if there is
            # more than one mention of the same entity then only the last is kept.
            entity_map = dict()
            item['golden-entity-mentions'] = []
            item['golden-event-mentions'] = []

            # Include any entity mentions whose bounds are strictly inside the sentence.
            for entity_mention in self.entity_mentions:
                entity_position = entity_mention['position']

                if text_position[0] <= entity_position[0] and entity_position[1] <= text_position[1]:

                    item['golden-entity-mentions'].append({
                        'text': self.clean_text(entity_mention['text']),
                        'position': entity_position,
                        'entity-type': entity_mention['entity-type'],
                        "entity_id": entity_mention['entity-id']
                    })
                    entity_map[entity_mention['entity-id']] = entity_mention

            # Include any event mentions whose bounds are strictly inside the sentence.
            #
            # TODO This might have to change for RichERE -- we might want to allow events not
            #  mentioned directly in the sentence? I am not sure if this is a thing.
            for event_mention in self.event_mentions:
                event_position = event_mention['trigger']['position']
                if text_position[0] <= event_position[0] and event_position[1] <= text_position[1]:
                    event_arguments = []
                    # TODO I will *probably* want to make sure event arguments get appended to entity
                    #  mentions even when they're not strictly in the sentence
                    for argument in event_mention['arguments']:
                        try:
                            entity_type = entity_map[argument['entity-id']]['entity-type']
                        except KeyError:
                            logger.warning('The entity in the other sentence is mentioned. This argument will be ignored.')
                            continue

                        event_arguments.append({
                            'role': argument['role'],
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

    # TODO needed for Rich ERE?
    def find_correct_offset(self, document_text: str, start_index: int, text: str) -> int:
        """
        Given the document text, a start index, and some text coming from the document,
        find the smallest offset that will make the start index correct
        as an index into the document text.

        Smallest here means smallest in absolute value. So we try 0 first, then +/-1, then +/-2,
        etc. The maximum offset that will be tried is 10000 characters.
        """
        offset = 0
        for i in range(0, 10000):
            for j in [-1, 1]:
                offset = i * j
                if document_text[start_index + offset:start_index + offset + len(text)] == text:
                    return offset

        logger.warning('[Warning] fail to find offset! (start_index: {}, text: {}, path: {})'.format(start_index, text, self.path))
        return offset

    def fix_wrong_position(self):
        """
        Loop over the mentions parsed and fix up their positions.

        This means finding the correct offset for each mention and adding the offfset to both
        components of that mention's position.
        """
        for entity_mention in self.entity_mentions:
            offset = self.find_correct_offset(
                document_text=self.document_text,
                start_index=entity_mention['position'][0],
                text=entity_mention['text'])

            entity_mention['position'][0] += offset
            entity_mention['position'][1] += offset

        for event_mention in self.event_mentions:
            offset1 = self.find_correct_offset(
                document_text=self.document_text,
                start_index=event_mention['trigger']['position'][0],
                text=event_mention['trigger']['text'])
            event_mention['trigger']['position'][0] += offset1
            event_mention['trigger']['position'][1] += offset1

    def parse_document(self, document_path):
        with open(document_path, 'r') as f:
            soup = BeautifulSoup(f.read(), features='html.parser')
            self.document_text = self._get_document_text(soup)

            sents = []
            converted_text = soup.text

            for sent in nltk.sent_tokenize(converted_text):
                sents.extend(sent.split('\n\n'))
            sents = [x for x in sents if len(x) > 5]
            # Do we *want* to drop the first one in Rihc?
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

    @staticmethod
    def _get_document_text(soup):
        return ''.join(Parser._force_include_author(string) for string in soup.strings)

    @staticmethod
    def _force_include_author(string: NavigableString) -> str:
        strings = []
        attrs = string.parent.attrs

        author = attrs.get('author')
        if author is not None:
            strings.append(f'By: {author}')

        original_author = attrs.get('original_author')
        if original_author is not None:
            strings.append(f'Quote by {original_author}:')

        strings.append(string)
        return ''.join(strings)

    def parse_annotations(self, xml_path):
        entity_mentions, event_mentions = [], []
        tree = ElementTree.parse(xml_path)
        root = tree.getroot()

        def found_entity(entity):
            entity_mentions.extend(self.parse_entity_tag(entity))

        def found_hopper(hopper):
            event_mentions.extend(self.parse_hopper_tag(hopper))

        def found_relation(relation):
            # TODO handle properly
            pass

        def found_filler(filler):
            # TODO handle properly
            pass

        self._parse_annotation_type(root, 'entities', 'entity', found_entity)
        self._parse_annotation_type(root, 'fillers', 'filler', found_filler)
        self._parse_annotation_type(root, 'relations', 'relation', found_relation)
        self._parse_annotation_type(root, 'hoppers', 'hopper', found_hopper)

        return entity_mentions, event_mentions

    # TODO is this actually useful?
    @staticmethod
    def _parse_annotation_type(root, annotations_type, child_annotation_tag, tag_found_callback):
        annotations = root.find(annotations_type)
        for child in annotations:
            if child.tag == child_annotation_tag:
                tag_found_callback(child)
            else:
                raise RuntimeError(f'While parsing {annotations_type}, found unexpected child tag {child.tag}')

    _NOUN_TYPE_TO_GLOSS: Mapping[str, str] = {
        "NAM": "name",
        # TODO correct gloss?
        "NOM": "nominal-phrase",
        "PRO": "pronoun",
    }

    @staticmethod
    def parse_entity_tag(node):
        entity_mentions = []

        for child in node:
            if child.tag == 'entity_mention':
                mention_text = child[0]
                assert mention_text.text.strip() == mention_text.text
                assert mention_text.tag == 'mention_text'

                start = int(child.attrib['offset'])
                length = int(child.attrib['length'])
                end = start + length

                # Strip a trailing quote, because entity names shouldn't have quotes in them.
                if mention_text.text[-1] == '"':
                    end -= 1
                    mention_text.text = mention_text.text[:-1]

                entity_mention = dict()
                entity_mention['entity-id'] = node.attrib['id']
                entity_type = (
                    '{}:{}'.format(node.attrib['type'], node.attrib['subtype'])
                    if 'subtype' in node.attrib
                    else node.attrib['type']
                )
                entity_mention['entity-type'] = entity_type
                entity_mention['noun-type'] = Parser._NOUN_TYPE_TO_GLOSS[child.attrib['noun_type']]
                entity_mention['specificity'] = node.attrib['specificity']
                entity_mention['entity-mention-id'] = child.attrib['id']
                entity_mention['source'] = child.attrib['source']
                entity_mention['text'] = mention_text.text
                entity_mention['position'] = [start, end]

                entity_mentions.append(entity_mention)
            else:
                raise RuntimeError(f'While parsing entity, found unexpected child {child.tag}.')

        return entity_mentions

    @staticmethod
    def parse_hopper_tag(node):
        event_mentions = []
        for child in node:
            if child.tag == 'event_mention':
                event_mention = dict()
                event_mention['event_type'] = '{}:{}'.format(child.attrib['type'], child.attrib['subtype'])
                event_mention['realis'] = child.attrib['realis']
                event_mention['arguments'] = []
                for child2 in child:
                    if child2.tag == 'trigger':
                        start = int(child2.attrib['offset'])
                        length = int(child2.attrib['length'])
                        end = start + length
                        event_mention['trigger'] = {
                            'text': child2.text,
                            'position': [start, end],
                        }
                    elif child2.tag == 'em_arg':
                        # There are two cases: Either the argument is an entity or it's a filler
                        argument = {
                            'text': child2.text,
                            'role': child2.attrib['role'],
                            'realis': child2.attrib['realis'],
                        }
                        if 'entity_id' in child2.attrib:
                            # TODO position not explicitly given, have to figure out from the entity mention ids
                            argument.update({
                                'entity-id': child2.attrib['entity_id'],
                                'entity-mention-id': child2.attrib['entity_mention_id'],
                            })
                        elif 'filler_id' in child2.attrib:
                            # TODO position not explicitly given, have to figure out from the filler ids?
                            argument.update({
                                'filler-id': child2.attrib['filler_id'],
                            })
                        else:
                            raise RuntimeError(f'Got em_arg {child2}, but it is neither an entity nor a filler.')

                        event_mention['arguments'].append(argument)
                    else:
                        raise RuntimeError(f'While parsing event mention, found unexpected child tag {child2.tag}.')
                event_mentions.append(event_mention)
            else:
                raise RuntimeError(f'While parsing event, found unexpected child tag {child2.tag}.')
        return event_mentions

    # TODO Does Rich ERE have value/timex tags? I think it doesn't.
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
