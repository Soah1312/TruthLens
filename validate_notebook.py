import json

nb_path = 'TruthLens_Experiment_4_DeBERTa_v3_Base.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

print('=== NOTEBOOK VALIDATION AUDIT ===')
print(f'nbformat: {nb.get("nbformat")}.{nb.get("nbformat_minor")}')
print(f'Total cells: {len(nb["cells"])}')

syntax_errors = 0
code_cells = 0
markdown_cells = 0

for idx, cell in enumerate(nb['cells']):
    ctype = cell.get('cell_type')
    assert ctype in ('markdown', 'code'), f'Invalid cell_type: {ctype} at cell {idx}'
    assert 'source' in cell, f'Missing source at cell {idx}'
    
    if ctype == 'markdown':
        markdown_cells += 1
    elif ctype == 'code':
        code_cells += 1
        assert 'outputs' in cell, f'Missing outputs at cell {idx}'
        assert 'execution_count' in cell, f'Missing execution_count at cell {idx}'
        
        # Test Python syntax (ignoring !shell commands)
        code_lines = [l for l in cell['source'] if not l.strip().startswith('!')]
        code = ''.join(code_lines)
        try:
            compile(code, f'<cell_{idx}>', 'exec')
        except SyntaxError as e:
            print(f'SYNTAX ERROR in Cell {idx}: {e}')
            syntax_errors += 1

print(f'Markdown cells: {markdown_cells}')
print(f'Code cells:     {code_cells}')
print(f'Syntax errors:  {syntax_errors}')
assert syntax_errors == 0, 'Syntax errors encountered'
print('SUCCESS: All code cells compile cleanly without syntax errors!')
