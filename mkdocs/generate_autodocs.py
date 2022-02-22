#!/usr/bin/env python

import pcot
import os
import shutil

template = """
# Autodocs
Below are automatically generated documents for certain entities
in PCOT. These are generated by running the *generate_autodocs.py*
script in the mkdocs directory.

## Nodes
Nodes are the entities which make up a PCOT document's graph, taking
inputs from various sources and manipulating them in various ways.

{nodes}

## *Expr* functions
Below are functions which can be used in the expression evaluation
node, [expr](expr).

{funcs}

## *Expr* properties
Below are properties which can be used in the expression evaluation
node, [expr](autodocs/expr).
Properties are names which can be used as identifiers on the
right hand side of a "." operator, such as *a.w* to get the width of an
image *a*.

{props}

"""

pcot.xform.createXFormTypeInstances()

parser = pcot.xform.allTypes['expr'].parser


if os.path.exists('docs/autodocs'):
    shutil.rmtree('docs/autodocs')

os.makedirs('docs/autodocs')

def genNodes():
    out = ""
    for realname, x in sorted(pcot.xform.allTypes.items()):
        name=realname.replace(' ', '_')
        out += f"* [{realname}]({name})\n"
        print(name)
        with open(f"docs/autodocs/{name}.md","w") as file:
            file.write(pcot.ui.help.getHelpMarkdown(x))
    return out
            
with open("docs/autodocs/index.md","w") as idxfile:
    str = template.format(nodes=genNodes(),
        funcs=parser.listFuncs(),
        props=parser.listProps())
    idxfile.write(str)
    
