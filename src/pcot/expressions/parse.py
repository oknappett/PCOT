# This is the core of the expression parsing system.
# The actual operations etc. are in eval.py.
import numbers
from io import BytesIO
from tokenize import tokenize, TokenInfo, NUMBER, NAME, OP, ENCODING, ENDMARKER, NEWLINE, ERRORTOKEN, PERCENT, DOT

from typing import List, Any, Optional, Callable, Dict, Tuple, Union

import pcot.conntypes as conntypes
from pcot.conntypes import Datum
from pcot.utils.html import HTML, Bold
from pcot.utils.table import Table

Stack = List[Any]

# will turn on printing
debug: bool = False


class ArgsException(Exception):
    ## @var message
    # a string message
    message: str

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class ParamException(Exception):
    ## @var message
    # a string message
    message: str

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class Parameter:
    """a definition of a function parameter"""

    def __init__(self,
                 name: str,  # name
                 desc: str,  # description
                 types: Union[conntypes.Type, Tuple[conntypes.Type, ...]],  # tuple of valid types, or just one
                 deflt: Optional[numbers.Number] = None  # default value for optional parameters, must be numeric
                 ):
        self.name = name
        if isinstance(types, conntypes.Type):
            self.types = (types,)  # convert single type to tuple
        else:
            self.types = types
        self.desc = desc
        self.deflt = deflt

    def isValid(self, datum: Datum):
        if datum is None:
            raise ParamException("None is not a valid parameter type")
        return datum.tp in self.types

    def getDefault(self):
        if self.deflt is None:
            raise ParamException("internal: optional parameter {} has a no default".format(self.name))
        return self.deflt

    def validArgsString(self):
        # turn a tuple (1,2,3) into "1, 2 or 3"
        if len(self.types) == 1:
            r = str(self.types[0])
        else:
            lst = list(self.types)
            last = lst.pop()
            r = "{} or {}".format(", ".join([str(x) for x in lst]), last)
        return r


def _styleTableCells(x):
    if x.tag in ('th', 'tr'):
        x.attrs = {"align": "left"}
    elif x.tag == "table":
        x.attrs = {"border": "1", "cellpadding": 5}


class Function:
    """a function callable from an eval string"""

    def __init__(self, name: str, fn: Callable[[List[Any]], Any], description: str,
                 mandatoryParams: List[Parameter], optParams: List[Parameter], varargs):
        self.fn = fn
        self.name = name
        self.desc = description
        self.mandatoryParams = mandatoryParams
        self.optParams = optParams
        self.varargs = varargs
        if self.varargs and len(self.mandatoryParams) == 0:
            raise ArgsException("cannot have a function with varargs but no mandatory arguments")
        elif self.varargs and len(self.optParams) > 0:
            raise ArgsException("cannot have a function with varargs and optional arguments")

    def help(self):
        t = Table()
        for x in self.mandatoryParams:
            t.newRow()
            t.add("name", x.name)
            t.add("types", x.validArgsString() + ("" if not self.varargs else "..."))
            t.add("description", x.desc)
        margs = t.htmlObj()
        t = Table()
        for x in self.optParams:
            t.newRow()
            t.add("name", x.name)
            t.add("types", x.validArgsString())
            t.add("description", x.desc)
        oargs = t.htmlObj()

        return HTML("div",
                    [
                        HTML("p", self.desc),
                        HTML("p", Bold("Mandatory arguments")),
                        margs,
                        HTML("p", Bold("Optional arguments")),
                        oargs
                    ]
                    ).visit(_styleTableCells).string()

    def chkargs(self, args: List[Optional[Datum]]):
        """Process arguments, returning a pair of lists of Datum items: mandatory and optional args.
        Takes two lists: one of tuples of permitted types of mandatory values (none of which can be None) and another of pairs of
        type tuple, defaultvalue for optional args"""
        mandatArgs = []
        optArgs = []
        lastparam = None
        try:
            if self.mandatoryParams is None:
                return args  # no type checking, just pass all args straight through
            for t in self.mandatoryParams:
                if len(args) == 0:
                    raise ArgsException('Not enough arguments in {}'.format(self.name))
                x = args.pop(0)
                if x is None:
                    raise ArgsException('None argument in {}'.format(self.name))
                elif t.isValid(x):
                    mandatArgs.append(x)
                else:
                    raise ArgsException(
                        'Bad argument in {}, got {}, expected {}'.format(self.name, x.tp, t.validArgsString()))
                lastparam = t

            if self.varargs:
                # varargs flag set - consume remaining args, using the last mandatory argument type
                for x in args:
                    if lastparam.isValid(t):
                        mandatArgs.append(x)
                    else:
                        raise ArgsException(
                            'Bad argument in {}, got {}, expected {}'.format(self.name, x.tp,
                                                                             lastparam.validArgsString()))

            for t in self.optParams:
                if len(args) == 0:
                    optArgs.append(Datum(conntypes.NUMBER, t.getDefault()))
                else:
                    x = args.pop(0)
                    if x is None:
                        raise ArgsException('None argument in {}'.format(self.name))
                    elif t.isValid(x):
                        optArgs.append(x)
                    else:
                        raise ArgsException(
                            'Bad argument in {}, got {}, expected {}'.format(self.name, x.tp, t.validArgsString()))
        except ParamException as e:
            # propagate parameter exception adding the name
            raise ArgsException("{}: {}".format(self.name, e.message))

        return mandatArgs, optArgs

    def call(self, args):
        # do type checking for arguments, then call. Note that we pass mandatory and optional arguments
        args, optargs = self.chkargs(args)
        return self.fn(args, optargs)


class Instruction:
    """Instruction interface"""

    def exec(self, stack: Stack):
        pass


class InstNumber(Instruction):
    val: float

    def __init__(self, v: float):
        self.val = v

    def exec(self, stack: Stack):
        # can't import NUMBER, it clashes with the one in tokenizer.
        stack.append(Datum(conntypes.NUMBER, self.val))

    def __str__(self):
        return "NUM {}".format(self.val)


class InstIdent(Instruction):
    val: str

    def __init__(self, v: str):
        self.val = v

    def exec(self, stack: Stack):
        stack.append(Datum(conntypes.IDENT, self.val))

    def __str__(self):
        return "STR {}".format(self.val)


class InstVar(Instruction):
    callback: Callable[[], Any]

    def __init__(self, name, callback: Callable[[], Any]):
        self.callback = callback
        self.name = name

    def exec(self, stack: Stack):
        stack.append(self.callback())

    def __str__(self):
        return "VAR {}".format(self.name)


class InstFunc(Instruction):
    callback: Callable[[List[Any]], Any]

    def __init__(self, name, func: Function):
        self.func = func
        self.name = name

    def exec(self, stack: Stack):
        # actually stack the callback function, don't call it - InstCall does that.
        stack.append(Datum(conntypes.FUNC, self.func))

    def __str__(self):
        return "FUNC {}".format(self.name)


class InstOp(Instruction):
    name: str
    prefix: bool
    precedence: int

    def __init__(self, n: str, pre: bool, parser: 'Parser'):
        if pre and n in parser.unopRegistry:
            self.precedence, self.callback = parser.unopRegistry[n]
        elif not pre and n in parser.binopRegistry:
            self.precedence, self.callback = parser.binopRegistry[n]
        else:
            raise ParseException("unknown {} operator: {}".format('prefix' if pre else 'suffix', n))
        self.name = n
        self.prefix = pre

    def exec(self, stack: Stack):
        if self.prefix:
            stack.append(self.callback(stack.pop()))
        else:
            b = stack.pop()
            a = stack.pop()
            stack.append(self.callback(a, b))

    def __str__(self):
        return "OP {} {}".format(self.name, "PRE" if self.prefix else "IN")


class InstBracket(InstOp):
    def __init__(self, parser: 'Parser'):
        # still need the string argument so InstOp knows which precedence to look up
        super().__init__('(', False, parser)
        self.argcount = 0

    def exec(self, stack: Stack):
        raise Exception("bracket instruction should never be executed")


class InstCall(Instruction):
    def __init__(self, argcount):
        self.argcount = argcount

    def __str__(self):
        return "CALL  argcount: {}".format(self.argcount)

    def exec(self, stack: Stack):
        # semantics here - pop off the args, then pop off the function name (or some other kind of ID!)
        if self.argcount != 0:
            args = stack[-self.argcount:]
        else:
            args = []
        # args.reverse()   # is this faster than just popping them in reverse order?
        # this is really annoying; can't delete multiple items from the stack without slicing and
        # can't slice because that wouldn't change the original stack.
        for x in range(0, self.argcount):
            stack.pop()
        v = stack.pop()
        if v.tp == conntypes.IDENT:
            raise ParseException("unknown function '{}' ".format(v.val))
        elif v.tp == conntypes.FUNC:
            # this executes the function by calling it's call method,
            # which will do argument type checking.
            stack.append(v.val.call(args))


class ParseException(Exception):
    def __init__(self, msg: str, t: Optional[TokenInfo] = None):
        if t is not None:
            msg = "{}: '{}' at chars {}-{}".format(msg, t.string, t.start[1], t.end[1])
        super().__init__(msg)


def execute(seq: List[Instruction], stack: Stack) -> float:
    for inst in seq:
        inst.exec(stack)
        if debug:
            print("EXECUTED {}, STACK NOW:".format(inst))
            for x in stack:
                print(x)
    return stack[0]


def isOp(t):
    return t.type in [OP, ERRORTOKEN, PERCENT, DOT]


class Parser:
    """Expression parser using the shunting algorithm, also incorporating the virtual machine for evaluation"""
    list: List[TokenInfo]
    output: List[Instruction]
    opstack: List[Instruction]  # compile stack

    ## if true, naked identifiers are stacked as strings.
    nakedIdents: bool

    ## binops are names (e.g. '+') mapped to precedence and two-arg fn which return a value
    binopRegistry: Dict[str, Tuple[int, Optional[Callable[[Any, Any], Any]]]]

    def registerBinop(self, name: str, precedence: int, fn: Callable[[Any, Any], Any]):
        """Register a binary operation"""
        self.binopRegistry[name] = (precedence, fn)

    ## unary ops are names mapped to precedence and single arg function which returns a value
    unopRegistry: Dict[str, Tuple[int, Optional[Callable[[Any], Any]]]]

    def registerUnop(self, name: str, precedence: int, fn: Callable[[Any], Any]):
        """Register a unary operation"""
        self.unopRegistry[name] = (precedence, fn)

    ## vars are names mapped to argless fns which return a value
    varRegistry: Dict[str, Callable[[], Any]]

    def registerVar(self, name: str, fn: Callable[[], Any]):
        """register a variable with a function to fetch it"""
        self.varRegistry[name] = fn

    ## other functions are names mapped to functions
    ## which take a list of args and return an arg
    funcRegistry: Dict[str, Function]

    def registerFunc(self, name: str,
                     description: str,
                     # this is a list of Parameter objects, one for each mandatory parameter. Default values on the parameters are ignored.
                     # if none, we don't type check at all.
                     mandatoryParams: Optional[List[Parameter]],
                     # This is a Parameter objects, one for each optional parameter, each of which should have a default.
                     optParams: List[Parameter],
                     # the actual function to call, which takes a list of mandatory arguments, a list of optional arguments,
                     # and returns a datum.
                     fn: Callable[[List[Datum], List[Datum]], Datum],
                     # if true, there are no optional arguments and all extra args must
                     # have the same type as the last mandatory argument
                     varargs=False,
                     ):
        """register a function - the callable should take a list of args, a list of optional args and return a value.
        Also takes a description and two lists of argument types: mandatory and optional."""
        self.funcRegistry[name] = Function(name, fn, description, mandatoryParams, optParams, varargs)

    def funcHelp(self, name):
        if name in self.funcRegistry:
            return self.funcRegistry[name].help()
        else:
            return "Function not found"

    def listFuncs(self):
        t = Table()
        for name, f in self.funcRegistry.items():
            t.newRow()
            t.add("name", name)
            ps = ",".join([p.name for p in f.mandatoryParams])
            if f.varargs:
                ps += "..."
            t.add("params", ps)
            t.add("opt. params", ",".join([p.name for p in f.optParams]))
            t.add("description", f.desc)
        return t.htmlObj().visit(_styleTableCells).string()

    def out(self, inst: Instruction):
        # internal method - output
        self.output.append(inst)

    def stackOp(self, op: InstOp):
        # internal method - stack operator
        self.opstack.append(op)

    def __init__(self, nakedIdents=False):
        """Initialise the parser, clearing all registered vars, funcs and ops."""
        self.binopRegistry = dict()
        self.unopRegistry = dict()
        self.varRegistry = dict()
        self.funcRegistry = dict()
        self.toks = []

        self.nakedIdents = nakedIdents

        # preregister the special operators for open bracket
        self.binopRegistry['('] = (100, None)
        self.unopRegistry['('] = (100, None)

    def parse(self, s: str):
        """Parsing function - uses the shunting algorithm.
        See also
        https://stackoverflow.com/questions/16380234/handling-extra-operators-in-shunting-yard/16392115
        """
        s = s.replace('\n', '').replace('\r', '')  # remove rogue newlines
        x = BytesIO(s.encode())
        self.toks = [x for x in (tokenize(x.readline) or []) if x.type != ENCODING
                     and x.type != ENDMARKER and
                     x.type != NEWLINE]
        self.opstack = []
        self.output = []

        wantOperand = True

        while True:
            if wantOperand:
                ######### In this mode, we want an operand next. Otherwise we want an operator.
                #   read a token. If there are no more tokens, announce an error.
                t = self.next()
                if t is None:
                    raise ParseException("premature end", t)
                elif isOp(t):
                    #  if the token is an prefix operator or an '(':
                    #    mark it as prefix and push it onto the operator stack
                    # Note that the bracket case is different from when we meet it in the
                    # !wantOperand case. It's just a prefix op here, otherwise we'd end up
                    # generating a CALL when we don't want one.
                    if t.string == '-' or t.string == '(':
                        self.stackOp(InstOp(t.string, True, self))
                        #     goto want_operand (we just loop)
                    elif t.string == ')':
                        #     if we meet a ')' it should be a function with no arguments
                        if len(self.opstack) == 0:  # we need that left bracket
                            raise ParseException("syntax error : bad no-arg function - no left bracket", t)
                        op = self.opstack.pop()
                        if isinstance(op, InstBracket) and not op.prefix:
                            if op.argcount != 0:
                                raise ParseException("syntax error : bad no-arg function - has args!", t)
                            self.out(InstCall(0))
                            wantOperand = False
                        else:
                            raise ParseException("syntax error : no arg-func with no name? ", t)
                #   if the token is an operand (identifier or variable):
                #     add it to the output queue
                #     goto have_operand
                elif t.type == NUMBER:
                    self.out(InstNumber(float(t.string)))
                    wantOperand = False
                elif t.type == NAME:
                    if t.string in self.varRegistry:
                        self.out(InstVar(t.string, self.varRegistry[t.string]))
                    elif t.string in self.funcRegistry:
                        fn = self.funcRegistry[t.string]
                        self.out(InstFunc(t.string, fn))
                    elif self.nakedIdents:
                        self.out(InstIdent(t.string))
                    else:
                        raise ParseException("unknown variable or function", t)
                    wantOperand = False
                else:
                    #   if the token is anything else, announce an error and stop.
                    raise ParseException("weird token", t)
            else:  # wantOperand = false
                ######### In this mode, we want an operator next. Otherwise we want an operand.
                #   read a token
                t = self.next()
                #   if there are no more tokens:
                if t is None:
                    #     pop all operators off the stack, adding each one to the output queue.
                    while len(self.opstack) > 0:
                        op = self.opstack.pop()
                        #     if a `(` is found on the stack, announce an error and stop.
                        if isinstance(op, InstOp) and op.name == '(':
                            raise ParseException("syntax error : mismatched bracket left", t)
                        self.out(op)
                    return  # ALL DONE
                #   if the token is a postfix operator:
                #   could deal with postfix here, but won't.

                # if the token is a ')':
                if isOp(t) and t.string == ')':
                    #     while the top of the stack is not '(':
                    while self.stackTopIsNotLPar():
                        #       pop an operator off the stack and add it to the output queue
                        op = self.opstack.pop()
                        self.out(op)
                    #     if the stack becomes empty, announce an error and stop.
                    if len(self.opstack) == 0:
                        raise ParseException("syntax error : mismatched bracket right", t)
                    #     if the '(' is marked infix, add a "call" operator to the output queue (*)
                    #     (using the arg count from the '(' )
                    op = self.opstack.pop()
                    if not op.prefix:
                        if not isinstance(op, InstBracket):
                            raise ParseException("syntax error : mismatched bracket right 2", t)
                        self.out(InstCall(op.argcount + 1))
                    #     pop the '(' off the top of the stack
                    #     goto have_operand (already there)

                elif isOp(t) and t.string == ',':
                    #   if the token is a ',':
                    #     while the top of the stack is not '(':
                    while self.stackTopIsNotLPar():
                        #       pop an operator off the stack and add it to the output queue
                        op = self.opstack.pop()
                        self.out(op)
                    # top of stack should now be a '(', which is special - increment the argument count
                    self.opstack[-1].argcount += 1
                    #                    self.out(InstComma())  # JCF mod - add comma as actual operator with low precendence
                    #     if the stack becomes empty, announce an error
                    if len(self.opstack) == 0:
                        raise ParseException("syntax error : mismatched bracket left 2", t)
                    #     goto want_operand
                    wantOperand = True
                elif isOp(t):
                    #   if the token is an infix operator:
                    if t.string == '(':
                        op = InstBracket(self)
                    else:
                        op = InstOp(t.string, False, self)
                    while self.stackTopIsOperatorPoppable(op):
                        self.out(self.opstack.pop())
                    self.stackOp(op)
                    wantOperand = True
                else:
                    # token is probably an operand.
                    raise ParseException("syntax error : unexpected operand", t)

    def stackTopIsNotLPar(self):
        # internal method - stack top is NOT an open bracket
        if len(self.opstack) == 0:
            return False
        op = self.opstack[-1]  # peek
        # must be an operator, and not a left-parenthesis
        return op.name != '('

    def stackTopIsOperatorPoppable(self, curop):
        # internal method - stack top is poppable to output
        if len(self.opstack) == 0:
            return False
        op = self.opstack[-1]  # peek
        # must not be a left-parenthesis
        if op.name == '(':
            return False
        # operator at stack top must have greater precedence, or the same precedence and token is left-assoc
        # (which they all are)
        return op.precedence >= curop.precedence

    def next(self) -> TokenInfo:
        # internal method - get next token
        if self.toksLeft():
            if debug:
                print(self.toks[0])
            return self.toks.pop(0)
        else:
            return None

    def rewind(self, tok: TokenInfo):
        # rewinder, unused.
        self.toks.insert(0, tok)

    def peek(self) -> TokenInfo:
        # stack peek
        if self.toksLeft():
            return self.toks[0]
        else:
            return None

    def toksLeft(self) -> bool:
        # count remaining tokens
        return len(self.toks) > 0
