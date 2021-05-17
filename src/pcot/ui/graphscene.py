## @package graphscene
# This package deals with constructing and modifying the Qt Graphics View scene
# which represents the objects in an XFormGraph.


import math
from typing import List, Optional

from PyQt5 import QtWidgets, QtGui
from PyQt5.QtCore import Qt, QPointF
from PyQt5.QtGui import QColor, QFont, QTransform

import pcot.conntypes as conntypes
import pcot.ui as ui
import pcot.ui.namedialog
import pcot.utils.deb
import pcot.xform as xform

## do we have the Grandalf package for automatic graph layout?
hasGrandalf = True
try:
    # we might have grandalf, but probably not.
    from grandalf.layouts import SugiyamaLayout, VertexViewer
    from grandalf.graphs import *

    print("Grandalf is present and will be used for autolayout.")
    hasGrandalf = True
except ImportError:
    print("Grandalf is not present, autolayout will be awful.")


    # dummy class defs for when grandalf isn't present, to avoid rogue errors in type checking

    class Graph:
        pass


    class Vertex:
        pass


    class Edge:
        pass


    hasGrandalf = False

## the font we use for most things
mainFont = QFont()
mainFont.setFamily('Sans Serif')
mainFont.setPixelSize(10)

## the font for error codes
errorFont = QFont()
errorFont.setFamily('Sans Serif')
errorFont.setBold(True)
errorFont.setPixelSize(12)

# constants for node drawing

## width of a node in pixels
NODEWIDTH = 100
## height of a node in pixels (including connectors)
NODEHEIGHT = 50

## offset of xform name text into box
XTEXTOFFSET = 5
## offset of xform name text into box
YTEXTOFFSET = 5

## offset of error code
YERROROFFSET = 10
XERROROFFSET = 20

## height of connector boxes
CONNECTORHEIGHT = 10
## additional space between nodes in autolayouts
YPADDING = 10

## x offset of connector label
CONNECTORTEXTXOFF = 2
## y offset of connector label on inputs
INCONNECTORTEXTYOFF = -14
## y offset of connector label on outputs
OUTCONNECTORTEXTYOFF = 10
## length of arrowhead lines
ARROWHEADLENGTH = 10
## angle between arrowhead lines and main line
ARROWHEADANGLE = math.radians(15)

## size of help box at top-right of main box
HELPBOXSIZE = 10

## x,y offset for pasted copies of nodes
PASTEOFFSET = 20


# basic shapes with extra data attached so we can get the node

## Help box. This has no functionality, we can't make it catch clicks unless we make it
# selectable (which we don't want). That has to be done in the GMainRect parent.
class GHelpRect(QtWidgets.QGraphicsRectItem):
    def __init__(self, x, y, node, parent):
        super().__init__(x + node.w - CONNECTORHEIGHT, y, HELPBOXSIZE, HELPBOXSIZE, parent=parent)
        self.setBrush(Qt.blue)


## text in middle of main rect and on connections
class GText(QtWidgets.QGraphicsSimpleTextItem):
    def __init__(self, parent, text, node):
        super().__init__(text, parent=parent)
        self.node = node

    def setColour(self, col):
        self.setBrush(col)


## core rectangle for a node, is parent of connectors and texts (and possibly other things too)
class GMainRect(QtWidgets.QGraphicsRectItem):
    # x,y,w,h are inherited
    # offset from original position x,y
    offsetx: int
    offsety: int
    helprect: GHelpRect  # help rectangle (top-right corner)
    node: xform.XForm  # node to which I refer
    text: GText  # text field

    def __init__(self, x1, y1, w, h, node):
        self.offsetx = 0  # these are the distances from our original pos.
        self.offsety = 0
        super().__init__(x1, y1, w, h)
        self.setFlags(QtWidgets.QGraphicsItem.ItemIsSelectable |
                      QtWidgets.QGraphicsItem.ItemIsMovable |
                      QtWidgets.QGraphicsItem.ItemSendsGeometryChanges)
        self.node = node
        # and add in the help box
        self.helprect = GHelpRect(x1, y1, node, self)

    ## deal with items moving
    def itemChange(self, change, value):
        if change == QtWidgets.QGraphicsItem.ItemPositionChange:
            dx = value.x() - self.offsetx
            dy = value.y() - self.offsety
            self.offsetx = value.x()
            self.offsety = value.y()
            # update the node position; reconstructing the scene should work
            x, y = self.node.xy
            self.node.xy = (x + dx, y + dy)

            # remake all the scene's connection arrows, which is rather inefficient!
            self.scene().rebuildArrows()
        return super().itemChange(change, value)

    ## double click should find or open a tab, even to the extent
    # of focussing another window (an expanded tab); an action
    # delegated to the main window
    def mouseDoubleClickEvent(self, event):
        w = getEventWindow(event)
        if self.helprect.boundingRect().contains(event.pos()):
            w.openHelp(self.node)
        else:
            w.openTab(self.node)

    ## context menu event on nodes
    def contextMenuEvent(self, event):
        m = QtWidgets.QMenu()
        rename = m.addAction("Rename")
        if self.node.type.hasEnable:
            togact = m.addAction("Disable" if self.node.enabled else "Enable")
        else:
            togact = None

        # only worth doing if there are menu items!
        if not m.isEmpty():
            action = m.exec_(event.screenPos())
            if action is None:
                return
            elif action == togact:
                self.node.setEnabled(not self.node.enabled)
            elif action == rename:
                changed, newname = pcot.ui.namedialog.do(self.node.displayName)
                if changed:
                    self.node.rename(newname)
                    ui.mainwindow.MainUI.rebuildAll()


## connection rectangles at top and bottom of node
class GConnectRect(QtWidgets.QGraphicsRectItem):
    isInput: bool  # true if this is an input
    node: xform.XForm  # the node I'm on
    index: int  # the index of the input/output
    name: str  # the name shown next to the rect (could be "")

    def isVariant(self):
        tp = self.node.getInputType(self.index) if self.isInput else self.node.getOutputType(self.index)
        return tp == conntypes.VARIANT

    ## construct, giving parent object (GMainRect), rectangle data, node data, input/output and index.
    def __init__(self, parent, x1, y1, x2, y2, node, isInput, index):
        super().__init__(x1, y1, x2, y2, parent=parent)
        self.isInput = isInput
        self.index = index
        self.node = node
        self.name = self.typeChanged()

    ## called when the type of the connector changes, returns the connector name
    def typeChanged(self):
        index = self.index
        node = self.node
        if self.isInput:
            # ugly stuff.
            name, typename, desc = (
                node.type.inputConnectors[index][0], node.getInputType(index), node.type.inputConnectors[index][2])
        else:
            name, typename = (node.type.outputConnectors[index][0], node.getOutputType(index))
        brush = conntypes.getBrush(typename)
        t = QTransform().translate(self.rect().x(), 0)  # need to translate brush patterns
        brush.setTransform(t)
        self.setBrush(brush)
        return name

    ## Called when the mouse is clicked on a connector.
    # We match the event to a connection arrow (which contains all the details)
    # and start dragging the arrow. We only modify the underlying graph
    # when we release the mouse, to delete the underlying connection
    # and make a new one if we landed on a connection box. We should also
    # check for if there is no effective change.
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            if self.isVariant():  # can't connect variant connectors
                return

            # are we connected? Mostly we can only drag inputs (arrow heads); some of the rest
            # of the code might contradict this because we used to be able to drag outputs
            # too (until I realised an output might have more than one connection coming from it).
            # However, this code is still needed because we can drag an unconnected output
            # to an input.
            if self.isInput and self.node.inputs[self.index] is not None:
                # find the arrow; this is a bit ugly.
                arrow = None
                for a in self.scene().arrows:
                    if a.n2 == self.node and a.input == self.index:
                        arrow = a
                if arrow is not None:
                    # and set that arrow to be dragging (note the inverted self.isInput)
                    self.scene().startDraggingArrow(arrow, not self.isInput, event)
            else:
                # here we're trying to create a brand new connection. Create a "dummy"
                # arrow which has one end of the connection set to (None,0) and add it
                # to the scene and arrow list as usual.
                x = event.scenePos().x()
                y = event.scenePos().y()
                if self.isInput:
                    arrow = GArrow(x, y, x, y, None, 0, self.node, self.index)
                else:
                    arrow = GArrow(x, y, x, y, self.node, self.index, None, 0)
                self.scene().addItem(arrow)
                self.scene().arrows.append(arrow)
                # and set that arrow to be dragging (note the NOT inverted self.isInput)
                self.scene().startDraggingArrow(arrow, self.isInput, event)
        return super().mousePressEvent(event)


## a line with an arrow on the end, connecting two nodes
class GArrow(QtWidgets.QGraphicsLineItem):
    ## @var n1
    # "from" xform node
    n1: xform.XForm
    ## @var n2
    # "to" xform node
    n2: xform.XForm
    ## @var output
    # index of output in n1
    output: int
    ## @var input
    # index of input in n2
    input: int
    ## @var head
    # the arrowhead shape
    head: Optional[QtWidgets.QGraphicsPolygonItem]

    ## Constructor - keep track of the nodes and connection indices
    def __init__(self, x1, y1, x2, y2, n1, output, n2, inp):
        self.n1 = n1
        self.output = output
        self.n2 = n2
        self.input = inp
        super().__init__(x1, y1, x2, y2)
        self.head = None
        self.makeHead()

    ## convert to string (debugging and dumping)
    def __str__(self):
        name1 = "??" if self.n1 is None else self.n1.name
        name2 = "??" if self.n2 is None else self.n2.name
        return "{}/{} -> {}/{}".format(name1, self.output, name2, self.input)

    ## make the arrowhead as a Path child of this line item, deleting any old one
    def makeHead(self):
        if self.head is not None:
            self.head.setParentItem(None)  # not removeItem(), apparently
        line = self.line()
        vec = QtGui.QVector2D(line.p1() - line.p2()).normalized()
        x = vec.x()
        y = vec.y()
        cs = math.cos(ARROWHEADANGLE)
        sn = math.sin(ARROWHEADANGLE)
        xa = x * cs - y * sn
        ya = x * sn + y * cs
        xb = x * cs + y * sn
        yb = -x * sn + y * cs
        poly = QtGui.QPolygonF()
        x2 = line.p2().x()
        y2 = line.p2().y()
        poly << QPointF(x2, y2) << QPointF(x2 + xa * ARROWHEADLENGTH, y2 + ya * ARROWHEADLENGTH) << \
        QPointF(x2 + xb * ARROWHEADLENGTH, y2 + yb * ARROWHEADLENGTH)
        self.head = QtWidgets.QGraphicsPolygonItem(poly, parent=self)
        self.head.setBrush(Qt.black)

    ## whenever we change the line, rebuild the head
    def setLine(self, line):
        super().setLine(line)
        self.makeHead()


## given some kind of QGraphicsSceneEvent, get the containing view.
# We just walk up the receiving widget's tree until we find a view.
def getEventView(evt):
    w = evt.widget()
    while w is not None and not isinstance(w, QtWidgets.QGraphicsView):
        w = w.parent()
    if w is None:
        raise Exception("Cannot get scene event's view")
    return w


## Get the mainwindow associated with an event.
def getEventWindow(evt):
    return getEventView(evt).window


## add connector graphics to a node, assuming its n.rect has already been created.
def makeConnectors(n, x, y):
    if len(n.inputs) > 0:
        size = n.w / len(n.inputs)
        xx = x
        for i in range(0, len(n.inputs)):
            # connection rectangles are parented to the main rectangle
            r = GConnectRect(n.rect, xx, y, size, CONNECTORHEIGHT, n, True, i)
            text = GText(n.rect, r.name, n)
            text.setPos(xx + CONNECTORTEXTXOFF, y + INCONNECTORTEXTYOFF)
            text.setFont(mainFont)
            text.setZValue(1)
            n.inrects[i] = r
            xx += size
    nouts = len(n.type.outputConnectors)
    if nouts > 0:
        size = n.w / nouts
        xx = x
        for i in range(0, nouts):
            # connection rectangles are parented to the main rectangle
            yy = y + n.h - YPADDING - CONNECTORHEIGHT
            r = GConnectRect(n.rect, xx, yy, size, CONNECTORHEIGHT, n, False, i)
            text = GText(n.rect, r.name, n)
            text.setPos(xx + CONNECTORTEXTXOFF, yy + OUTCONNECTORTEXTYOFF)
            text.setFont(mainFont)
            text.setZValue(1)
            n.outrects[i] = r
            xx += size


## create the necessary items to create a node
## (assuming place has been called). Just the node, not the connections.
def makeNodeGraphics(n):
    x, y = n.xy
    # if the node doesn't have width and height yet, set them. This happens
    # when a node is not created in place().
    if n.w is None:
        n.w = NODEWIDTH
        n.h = NODEHEIGHT + YPADDING

    # draw basic rect, leaving room for connectors at top and bottom
    # We keep this rectangle in the node so we can change its colour
    n.rect = GMainRect(x, y + CONNECTORHEIGHT, n.w, n.h - YPADDING - CONNECTORHEIGHT * 2, n)

    # draw text label, using the display name. Need to keep a handle on the text
    # so we can change the colour in setColourToState(). This is drawn using a method
    # in the type, which we can override if we want to do Odd Things (editable text)
    n.rect.text = n.type.buildText(n)

    makeConnectors(n, x, y)

    # if there's an error, add the code. Otherwise if there a rect text, add that.
    if n.error is not None:
        error = GText(n.rect, "err:" + n.error.code, n)
        error.setFont(errorFont)
        error.setBrush(QColor(255, 0, 0))
        error.setPos(x + XTEXTOFFSET + XERROROFFSET, y + YTEXTOFFSET + CONNECTORHEIGHT + YERROROFFSET)
    elif n.rectText is not None:
        t = GText(n.rect, n.rectText, n)
        t.setFont(errorFont)
        t.setBrush(QColor(0, 0, 255))
        t.setPos(x + XTEXTOFFSET, y + YTEXTOFFSET + CONNECTORHEIGHT + YERROROFFSET)

    return n.rect


## The custom scene. Note that when serializing the nodes,
# the geometry fields should be dealt with.

class XFormGraphScene(QtWidgets.QGraphicsScene):
    ## the graph I represent
    graph: xform.XFormGraph
    ## selected nodes
    selection: List[xform.XForm]
    ## check selected nodes if the selection changes
    checkSelChange: bool
    ## list of arrows
    arrows: List[GArrow]
    ## dragging arrow or none if no arrow being dragged
    draggingArrow: Optional[GArrow]

    ## initialise to a graph, and do autolayout if doPlace is true
    def __init__(self, graph, doPlace):
        super().__init__()
        self.graph = graph
        self.selectionChanged.connect(self.selChanged)
        self.selection = []
        self.checkSelChange = True
        # place everything, adding xy,w,h to all nodes
        if doPlace:
            self.place()
        else:
            # or alternatively just set w,h (for loading data from a file)
            for n in self.graph.nodes:
                n.w = NODEWIDTH
                n.h = NODEHEIGHT + YPADDING

        # and make all the graphics
        self.rebuild()

    ## try to autolayout a graph using Grandalf (or at least x,y coordinates inside
    # the xforms).

    def placeGrandalf(self):
        if len(self.graph.nodes) == 0:  # no nodes to place
            return

        g = Graph()  # grandalf graph, not one of ours!
        # add the vertices
        for n in self.graph.nodes:
            n.vert = Vertex(n)
            n.vert.view = VertexViewer(w=NODEWIDTH, h=NODEHEIGHT + YPADDING)
            g.add_vertex(n.vert)
        # now the edges
        for n in self.graph.nodes:
            for ii in range(0, len(n.inputs)):
                inp = n.inputs[ii]
                if inp is not None:
                    other, output = inp
                    n1 = n.vert
                    n2 = other.vert
                    g.add_edge(Edge(n1, n2))
        # build the layout separately for each unconnected
        # subgraph:

        for gr in g.C:
            sug = SugiyamaLayout(gr)
            sug.init_all()
            sug.draw(3)  # 3 iterations of algorithm

        # invert y coordinates of nodes so we have the source at the top,
        # and copy into geometry into the node (if we don't use grandalf)
        cy = max([n.vert.view.xy[1] for n in self.graph.nodes]) / 2
        for n in self.graph.nodes:
            x, y = n.vert.view.xy
            n.w = n.vert.view.w
            n.h = n.vert.view.h
            n.xy = (x, cy - y)

    ## autolayout a graph, using Grandalf if available or something dumb if it isn't    
    def place(self):
        if hasGrandalf:
            self.placeGrandalf()
        else:
            # this is what we do if we don't have Grandalf
            x = 0
            y = 0
            for n in self.graph.nodes:
                n.w = NODEWIDTH
                n.h = NODEHEIGHT + YPADDING
                n.xy = (x, y)
                y += n.h + 20

    ## rebuild the entire scene from the graph
    def rebuild(self):
        # this will (obv.) change the rubberband selection, so you might get a crash
        # if you do this with live objects; we clear the selection to avoid this
        self.checkSelChange = False
        self.clearSelection()
        self.clear()
        self.checkSelChange = True
        # create the graphics items
        for n in self.graph.nodes:
            # makes the graphics for the node
            gfx = makeNodeGraphics(n)
            self.addItem(gfx)
            # no tab to front by default
        # and make the arrows
        self.arrows = []
        self.draggingArrow = None
        self.rebuildArrows()
        self.selChanged()

    ## return a good position for a new item placed with no hint to where it should go    
    def getNewPosition(self):
        if len(self.graph.nodes) > 0:
            xs = [n.xy[0] for n in self.graph.nodes]
            ys = [n.xy[1] for n in self.graph.nodes]
            x = sum(xs) / len(xs)
            y = max(ys) + NODEHEIGHT
            return x, y
        else:
            return 0, 0

    ## assuming that all the nodes have been placed and makeNodeGraphics has been called,
    ## connect them all up with arrows according to the connections.
    def rebuildArrows(self):
        # get rid of old arrows
        for line in self.arrows:
            self.removeItem(line)
        self.arrows = []
        for n2 in self.graph.nodes:  # n2 is the destination node
            for inputIdx in range(0, len(n2.inputs)):
                inp = n2.inputs[inputIdx]
                if inp is not None:
                    n1, output = inp  # n1 is the source node

                    x1, y1 = n1.xy  # this is the "from" and should be on the output ctor
                    x2, y2 = n2.xy  # this is the "to" and should be on the input ctor
                    # draw lines
                    outsize = n1.w / len(n1.type.outputConnectors)
                    insize = n2.w / len(n2.inputs)
                    x1 = x1 + outsize * (output + 0.5)
                    x2 = x2 + insize * (inputIdx + 0.5)
                    y1 += n1.h - YPADDING
                    arrowItem = GArrow(x1, y1, x2, y2, n1, output, n2, inputIdx)
                    self.addItem(arrowItem)
                    self.arrows.append(arrowItem)

    ## handle selection and other state by changing the colour of the main rect of the selected item
    # Now this is a bit complex because we have to show a colour change for the error state and
    # there are two selections - the selected items in the view (selected by clicking
    # and rubberband) and the currently shown tab.
    def setColourToState(self):
        for n in self.graph.nodes:
            r = 255
            g = 255
            b = 255
            if n in self.selection:
                r -= 50
                g -= 50
            if n.current:
                r -= 50
                b -= 50
            if n.error is not None:
                r = 255
                g -= 50
                b -= 50
            if n.rect is not None:  # might not have a brush yet (rebuild might need calling)
                n.rect.setBrush(QColor(r, g, b))
                outlinecol = QColor(0, 0, 0) if n.enabled else QColor(255, 0, 0)
                n.rect.setPen(outlinecol)
                n.rect.text.setColour(outlinecol)
        self.update()

    ## handle the selection area being changed (this is a UI slot)
    def selChanged(self):
        if self.checkSelChange:
            items = self.selectedItems()
            self.selection = []
            for n in self.graph.nodes:
                if n.rect in items:
                    self.selection.append(n)
        self.setColourToState()

    ## current tab has changed, set up the UI accordingly. May be passed None if all tabs
    # are closed!
    def currentChanged(self, node):
        for n in self.graph.nodes:
            n.current = n is node
        self.selChanged()

    ## start dragging an arrow - draggingArrowStart indicates whether we are
    # dragging the head (false) or tail (true). The event is the event which
    # triggered the drag - a mouse press on a connection.
    def startDraggingArrow(self, arrow, draggingArrowStart, event):
        self.draggingArrowStart = draggingArrowStart
        self.draggingArrow = arrow
        self.dragStartPos = event.pos()
        # temporarily disable drag-selection
        v = getEventView(event)
        v.setDragMode(QtWidgets.QGraphicsView.NoDrag)

    ## here is where we handle actually dragging an arrow around. Dragging
    # items is managed by the QGraphicsView.    
    def mouseMoveEvent(self, event):
        if self.draggingArrow is not None:
            p = event.scenePos()
            line = self.draggingArrow.line()
            if self.draggingArrowStart:
                line.setP1(p)
            else:
                line.setP2(p)
            self.draggingArrow.setLine(line)
        super().mouseMoveEvent(event)

    ## handle releasing the mouse button during arrow dragging
    def mouseReleaseEvent(self, event):
        # first, go back to normal dragging
        v = getEventView(event)
        v.setDragMode(QtWidgets.QGraphicsView.RubberBandDrag)
        if self.draggingArrow is not None:
            arrow = self.draggingArrow  # disconnects seem to remove this?
            # first, make sure we close off the movement
            self.mouseMoveEvent(event)
            # get all the connectors at the event location. There should be one or none.
            x = [x for x in self.items(event.scenePos()) if isinstance(x, GConnectRect)]
            if not x:  # is empty? We are dragging to a place with no connector
                # if there is an existing connection we are deleting it
                if arrow.n2 is not None:  # if a connection exists and we are removing it
                    # remove the connection in the model (n2 might get changed)
                    arrow.n2.disconnect(arrow.input)
                    self.graph.inputChanged(arrow.n2)
            else:
                conn = x[0]  # this is the GConnectRect we are connecting to/from

                # get the connection data for the connection we want to make
                if self.draggingArrowStart:
                    # we are dragging an output, so we want to connect an input to
                    # this new output
                    n1 = conn.node
                    output = conn.index
                    n2 = arrow.n2
                    inputIdx = arrow.input
                else:
                    # we are dragging an input, so we want to connect an output to the new input
                    n2 = conn.node
                    inputIdx = conn.index
                    n1 = arrow.n1
                    output = arrow.output

                # are they compatible?
                outtype = n1.getOutputType(output)
                intype = n2.getInputType(inputIdx)

                if conntypes.isCompatibleConnection(outtype, intype):
                    if n2.cycle(n1):
                        ui.error("cannot create a cycle")
                    else:
                        # remove existing connections at the connector we are dragging to
                        # if it is an input
                        if conn.isInput:
                            conn.node.disconnect(conn.index)
                            self.graph.inputChanged(conn.node)
                        # We are dragging the connection to a new place.
                        # is it an existing connection we are modifying?
                        # The case where it's a fresh output being dragged to an input
                        # works too.
                        if arrow.n2 is not None:
                            # disconnect the existing connection
                            arrow.n2.disconnect(arrow.input)
                            self.graph.inputChanged(arrow.n2)
                        n2.connect(inputIdx, n1, output)
                        self.graph.inputChanged(n2)
                else:
                    ui.error("incompatible connection types OUT {} -> IN {}".format(outtype, intype))
            self.rebuildArrows()
            self.draggingArrow = None
        super().mouseReleaseEvent(event)

    ## copy operation, serialises the items to the system clipboard
    def copy(self):
        self.graph.copy(self.selection)
        pcot.utils.deb.show(self)

    ## paste operation, deserialises items from the system clipboard
    def paste(self):
        # clear the selection area (UI controlled)
        self.clearSelection()
        # paste the nodes, offset them, and select them.
        newnodes = self.graph.paste()
        # offset all the nodes
        for n in newnodes:
            x, y = n.xy
            n.xy = (x + PASTEOFFSET, y + PASTEOFFSET)
        self.rebuild()  # rebuild all nodes
        self.selection = newnodes  # set the selection to the new nodes
        for n in newnodes:
            n.rect.setSelected(True)
        # colour selected nodes
        self.setColourToState()

    ## cut operation, serialises items to system clipboard and deletes them
    def cut(self):
        self.graph.copy(self.selection)
        for n in self.selection:
            self.graph.remove(n)
        self.selection = []
        self.rebuild()