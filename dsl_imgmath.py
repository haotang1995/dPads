import dsl

DSL_DICT = {
    # ('atom', 'atom',): [dsl.AddFunction, dsl.MultiplyFunction, dsl.SubFunction,
    ('atom', 'atom',): [dsl.imgmath.ImgMathAddFunction, dsl.imgmath.ImgMathMultiplyFunction, dsl.imgmath.ImgMathSubFunction,
                        dsl.imgmath.ImgMathXSelection, dsl.imgmath.ImgMathYSelection, dsl.imgmath.ImgMathZSelection,],
    ('list', 'list',): [dsl.MapFunction,],
}

CUSTOM_EDGE_COSTS = {
    ('atom', 'atom') : {}
}

