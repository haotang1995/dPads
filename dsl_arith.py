import dsl

DSL_DICT = {
    # ('atom', 'atom', 1, 1,): [dsl.AddFunction, dsl.MultiplyFunction, dsl.SubFunction,],
    # ('atom', 'atom', 3, 1): [dsl.arith.ArithXSelection, dsl.arith.ArithYSelection, dsl.arith.ArithZSelection,],
    # ('atom', 'atom', 3, 1): [dsl.arith.XSelection, dsl.arith.YSelection, dsl.arith.ZSelection,],
    ('atom', 'atom',): [dsl.AddFunction, dsl.MultiplyFunction, dsl.SubFunction,
                        dsl.arith.XSelection, dsl.arith.YSelection, dsl.arith.ZSelection,],
    ('list', 'list',): [dsl.MapFunction,],
    # ('list', 'atom', 3, 3,): [dsl.arith.SqueezeList,],
    # ('list', 'atom', 3, 1,): [],
}

CUSTOM_EDGE_COSTS = {
    ('atom', 'atom') : {}
}

