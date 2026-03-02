"""Implements a parameter with automatic gradient tracking.
"""

import math
from collections.abc import Callable
from enum import Enum


class Operation(str, Enum):
    """An enumeration that showcases the operations that can lead to the
    creation of new parameters (used for gradient tracking and calculation).
    """

    MANUAL_CREATION = ''
    SUM = '+'
    MUL = '*'
    TANH = 'tanh'
    RELU = 'relu'
    POWER = 'pow'
    ABS = 'abs'


class Parameter:
    """A parameter with automatic gradient tracking.
    """

    def __init__(self, value: int | float,
                 label: str | None = None,
                 _prev: tuple['Parameter', ...] | None = None,
                 _op: str | Operation = Operation.MANUAL_CREATION) -> None:
        """Creates a new `Parameter` instance.

        :param value: The value of the parameter.
        will be extracted from it (ignoring user-provided args).
        :param label: The parameter's name (if any), defaults to `None`
        :param _prev: The previous parameters that led to the creation
        of the parameter (if any), defaults to `None`
        :param _op: The operation that led to the creation of the
        parameter (if any), defaults to `Operation.MANUAL_CREATION`
        """
        self.value = value
        self.label = label
        self._prev = set(_prev if _prev is not None else tuple())
        self._op = _op

        self.grad = 0.0
        self._backward: Callable[[], None] = lambda: None

    def __repr__(self) -> str:
        """Provides a string representation of the parameter object.

        :return: The parameter object as a string representation.
        """
        return f'Parameter({self.value})'

    def __add__(self, param: 'int | float | Parameter') -> 'Parameter':
        """Adds a given parameter to the current object.

        :param param: The parameter to be added.
        :return: A new parameter with the value of the sum.
        """
        input_param = (Parameter(param)
                       if not isinstance(param, Parameter) else param)
        out_param = Parameter(self.value + input_param.value,
                              _prev=(self, input_param), _op=Operation.SUM)

        def _backward():
            self.grad += out_param.grad
            input_param.grad += out_param.grad

        out_param._backward = _backward

        return out_param

    def __radd__(self, param: 'int | float | Parameter') -> 'Parameter':
        """Performs the addition of param + self.

        :param param: The value to be added with self.
        :return: A new parameter obtained from adding
        the parameter  with self.
        """
        return self + param

    def __neg__(self) -> 'Parameter':
        """Multiplies the current value by -1.

        :return: The negative value of the current parameter.
        """
        return self * -1

    def __sub__(self, param: 'int | float | Parameter') -> 'Parameter':
        """Subtracts param from self creating a new parameter.

        :param param: The parameter to be used for subtraction.
        :return: A new param with the value of self - param.
        """
        return self + (-param)

    def __rsub__(self, param: 'int | float | Parameter') -> 'Parameter':
        """Subtracts self from param creating a new parameter.

        :param param: The parameter from which self will be subtracted.
        :return: A new parameter with value param - self.
        """
        return (-self) + param

    def __mul__(self, param: 'int | float | Parameter') -> 'Parameter':
        """Multiplies a given parameter by the given object.

        :param param: The parameter to be multiplied with self.
        :return: A new parameter with the product result.
        """
        param = Parameter(param) if not isinstance(param, Parameter) else param
        out_param = Parameter(self.value * param.value,
                              _prev=(self, param), _op=Operation.MUL)

        def _backward():
            self.grad += param.value * out_param.grad
            param.grad += self.value * out_param.grad

        out_param._backward = _backward

        return out_param

    def __rmul__(self, other: 'int | float | Parameter') -> 'Parameter':
        """Implements the right-side parameter multiplication operation.

        :param other: The left-side operand of the product operation.
        :return: The parameter obtained from the product operation.
        """
        return self * other

    def __pow__(self, exp: int | float) -> 'Parameter':
        """Performs the power operation with the given exponent.

        :param exp: The exponent to be used in the pow operation.
        :return: A new parameter with grad tracking as obtained from the
        power operation.
        """
        out_param = Parameter(self.value ** exp, _prev=(self, ),
                              _op=(Operation.POWER))

        def _backward() -> None:
            self.grad += exp * (self.value ** (exp - 1)) * out_param.grad

        out_param._backward = _backward

        return out_param

    def __truediv__(self, param: 'int | float | Parameter') -> 'Parameter':
        """Implements the division operation with automatic gradient
        calculation.

        :param param: The parameter to be used as right operand of the
        division.
        :return: The new parameter obtained from the div operation.
        """
        return self * (param ** -1)

    def __rtruediv__(self, param: 'int | float | Parameter') -> 'Parameter':
        """Implements the division of of the given left operand with self.

        :param param: The left-side operand of the division.
        :return: A new parameter obtained from the division of a left operand
        with self (right-side operand).
        """
        left_operand = (Parameter(param) if not isinstance(param, Parameter)
                        else param)

        return left_operand / self

    def tanh(self) -> 'Parameter':
        """Applies the tanh operation on the given parameter, creating a new
        one as a result.

        :return: A parameter obtained after applying tanh to the object.
        """
        exp_value = math.e ** (2 * self.value)
        out_param = Parameter((exp_value - 1) / (exp_value + 1),
                              _prev=(self, ), _op=Operation.TANH)

        def _backward():
            self.grad += (1.0 - out_param.value ** 2) * out_param.grad

        out_param._backward = _backward

        return out_param

    def relu(self) -> 'Parameter':
        """Applies the ReLU operation on the given parameter, creating a
        new one as a result.

        :return: A new parameter with the value obtained after
        applying the ReLU operation.
        """
        out_param = Parameter(0 if self.value < 0 else self.value,
                              _prev=(self, ), _op=Operation.RELU)

        def _backward() -> None:
            self.grad += (out_param.value > 0) * out_param.grad

        out_param._backward = _backward

        return out_param

    def abs(self) -> 'Parameter':
        """Creates a new parameter with the absolute value of self.

        :return: A new param containing the absolute value of self.
        """
        out_param = Parameter(abs(self.value), _prev=(self, ),
                              _op=Operation.ABS)

        def _backward() -> None:
            sign = -1.0 if self.value < 0 else 1.0
            self.grad += sign * out_param.grad

        out_param._backward = _backward

        return out_param

    def backward(self) -> None:
        """Runs backpropagation, storing gradient values
        in all the parameters that affect the current object.

        NOTE: The backward propagation will be performed so that a
        node's backpropagation function gets called after the required
        backpropagations already occurred.
        """
        self.grad = 1.0

        topological_graph: list[Parameter] = []
        visited: set[Parameter] = set()

        def build_topological_graph(param: Parameter):
            if param in visited:
                return

            visited.add(param)
            for prev_param in param._prev:
                build_topological_graph(prev_param)

            topological_graph.append(param)

        build_topological_graph(self)
        for param in reversed(topological_graph):
            param._backward()
