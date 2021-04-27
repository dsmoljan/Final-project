import PIL.Image
import PIL.ImageFile
import xml.etree.ElementTree as ET
import zipfile

def read_ora(ora_file):
	"""
	Read an ORA file and convert it to a dict for easy manipulation
	"""
	# Normalize between different kind of parameters
	ora_filename = None
	if isinstance(ora_file, str):
		ora_filename = ora_file
		if not zipfile.is_zipfile(ora_file):
			raise Exception('"{}" is not an Open Raster file: not a ZIP file'.format(ora_filename))
	else:
		ora_filename = getattr(ora_file, 'name', '<data>')

	# Read the file
	result = {
		'version': '0.0.0',
		'w': 0, 'h': 0,
		'xres': 72, 'yres': 72,
		'root': {
			'type': 'stack',
			'childs': []
		}
	}
	with zipfile.ZipFile(ora_file, 'r') as ora_archive:
		#TODO check for presence and contents of 'mimetype' file

		# Read XML stack tree
		stack_tree = None
		with ora_archive.open('stack.xml') as stack_index:
			stack_tree = ET.fromstring(stack_index.read())

		if (stack_tree.tag) != 'image':
			raise Exception('"{}" is not an Open Raster file: stack.xml root element is not <image>'.format(ora_filename))

		# Convert stack tree to our dict format
		_merge_attributes(result, stack_tree, int_attributes = ['w', 'h', 'xres', 'yres'], forbidden_attributes = ['root'])

		stack_tree_root = None
		for elem in stack_tree:
			if stack_tree_root is not None:
				raise Exception('"{}" is not an Open Raster file: stack.xml\'s <image> must have only one child'.format(ora_filename))
			if elem.tag != 'stack':
				raise Exception('"{}" is not an Open Raster file: stack.xml\'s <image> child must be <stack>'.format(ora_filename))
			stack_tree_root = elem
		result['root'] = _parse_stack(stack_tree_root)

		# Load layer images
		_load_rasters(result['root'], ora_archive)

	return result

def write_ora(ora_file):
	"""
	Convert a dict to an ORA archive and write it to a file
	"""
	raise Exception('ora.write_ora not yet implemented')

def _load_rasters(stack, ora_archive):
	PIL.ImageFile.LOAD_TRUNCATED_IMAGES = True # Without that, some GIMP files cannot be read: https://github.com/python-pillow/Pillow/issues/3287

	for child in stack['childs']:
		if child['type'] == 'stack':
			_load_rasters(child, ora_archive)
		elif child['type'] == 'layer':
			raster = None
			with ora_archive.open(child['src']) as img_file:
				raster = PIL.Image.open(img_file)
				raster.load() # Read before closing the file. Pillow's doc proposed solution is to use an io.BytesIO object to act as a buffer, but would mean we hold compressed image data even after parsing it.
			child['raster'] = raster

def _merge_attributes(dest, elem, int_attributes = None, float_attributes = None, forbidden_attributes = None):
	for attrib_name in elem.attrib:
		if forbidden_attributes is not None and attrib_name in forbidden_attributes:
			raise Exception('attribute "{}" on <{}> is not supported: conflict with internal format'.format(attrib_name, elem.tag))
		if int_attributes is not None and attrib_name in int_attributes:
			dest[attrib_name] = int(elem.attrib[attrib_name])
		elif float_attributes is not None and attrib_name in float_attributes:
			dest[attrib_name] = float(elem.attrib[attrib_name])
		else:
			dest[attrib_name] = elem.attrib[attrib_name]

def _parse_stack(xml_stack):
	stack = {
		'type': 'stack',
		'childs': [],
	}

	_merge_attributes(stack, xml_stack, int_attributes = ['w', 'h', 'xres', 'yres'], forbidden_attributes = ['type', 'childs'])

	for elem in xml_stack:
		if elem.tag == 'stack':
			stack['childs'].append(_parse_stack(elem))
		elif elem.tag == 'layer':
			stack['childs'].append(_parse_layer(elem))
		elif elem.tag == 'text':
			stack['childs'].append(_parse_text(elem))
		else:
			raise Exception('Open Raster stack.xml parsing failed: unknown <stack> child "<{}>"'.format(elem.tag))

	return stack

def _parse_layer(xml_layer):
	layer = {
		'type': 'layer',
	}

	_merge_attributes(layer, xml_layer, int_attributes = ['x', 'y'], float_attributes = ['opacity'], forbidden_attributes = ['type'])

	return layer

def _parse_text(xml_text):
	text = {
		'type': 'text',
		'contents': ''
	}

	_merge_attributes(text, xml_text, int_attributes = ['x', 'y'], forbidden_attributes = ['type', 'contents'])
	text['contents'] = xml_text.text

	return text
