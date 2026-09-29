"""Tests for ghav.expr (GitHub expression subset)."""

from __future__ import annotations

import math

import pytest

from ghav import expr as X
from ghav.wir import TRUE, Expr


def lit(v):
    return Expr("lit", (v,))


def ctx(k):
    return Expr("ctx", (k,))


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def test_parse_status_and_context():
    e = X.parse("success() && github.event_name == 'push'")
    assert e == Expr("and", (X.SUCCESS, Expr("eq", (ctx("github.event_name"), lit("push")))))


def test_parse_unwraps_single_expression():
    e = X.parse("${{ steps.cache-npm.outputs.cache-hit != 'true' }}")
    assert e == Expr("ne", (Expr("step", ("cache-npm", "outputs.cache-hit")), lit("true")))


def test_parse_needs_and_step_fields():
    assert X.parse("needs.build.result") == Expr("needs", ("build", "result"))
    assert X.parse("needs.build.outputs.version") == Expr("needs", ("build", "outputs.version"))
    assert X.parse("steps.x.outcome") == Expr("step", ("x", "outcome"))


@pytest.mark.parametrize(
    "text,value",
    [
        ("true", True),
        ("false", False),
        ("null", None),
        ("42", 42),
        ("-9.5", -9.5),
        ("0xff", 255),
        ("2e3", 2000.0),
        ("'it''s'", "it's"),
        ("''", ""),
    ],
)
def test_parse_literals(text, value):
    assert X.parse(text) == lit(value)


def test_parse_nan_literal():
    e = X.parse("NaN")
    assert e.op == "lit" and math.isnan(e.args[0])


def test_parse_precedence():
    # ! binds tighter than ==, == tighter than &&, && tighter than ||
    e = X.parse("!a.b == 'x' || c.d && e.f")
    assert e.op == "or"
    assert e.args[0] == Expr("eq", (Expr("not", (Expr("opaque", ("a.b", "a.b")),)), lit("x")))
    assert e.args[1].op == "and"


def test_parse_function_names_are_case_insensitive():
    e = X.parse("startswith(github.ref, 'refs/tags/')")
    assert e == Expr("call", ("startsWith", ctx("github.ref"), lit("refs/tags/")))
    assert X.parse("Always()") == Expr("status", ("always",))


def test_parse_bracket_property_becomes_path():
    assert X.parse("matrix['node-version']") == ctx("matrix.node-version")
    assert X.parse("github.event.pull_request.labels.*.name") == ctx(
        "github.event.pull_request.labels.*.name"
    )


def test_comparison_is_opaque_subexpression_with_refs():
    e = X.parse("github.run_number > 5 && success()")
    assert e.op == "and"
    assert e.args[0] == Expr("opaque", ("github.run_number > 5", "github.run_number"))
    assert e.args[1] == X.SUCCESS


def test_needs_wildcard_is_opaque():
    e = X.parse("contains(needs.*.result, 'failure')")
    assert e == Expr("call", ("contains", Expr("opaque", ("needs.*.result", "needs.*.result")), lit("failure")))
    assert X.needs_refs(e) == frozenset({"*"})


def test_unknown_function_is_opaque():
    e = X.parse("frobnicate(matrix.x)")
    assert e.op == "opaque"
    assert e.args[1:] == ("matrix.x",)


@pytest.mark.parametrize(
    "text",
    [
        "always() && (a",
        "\"double\" == 'quotes'",
        "a ==",
        "",
        "}}",
        "steps.",
        "${{ unterminated",
        "@@@",
    ],
)
def test_parse_never_raises(text):
    e = X.parse(text)
    assert isinstance(e, Expr)


def test_syntax_error_opaque_keeps_refs_and_status():
    e = X.parse("always() && (github.ref")
    assert e.op == "opaque"
    assert e.args == ("always() && (github.ref", "github.ref")
    assert X.uses_status_function(e)


def test_parse_non_string_scalars():
    assert X.parse(False) == lit(False)
    assert X.parse(True) == lit(True)


def test_parse_mixed_template_in_if():
    e = X.parse("${{ a.b }} x")
    assert e.op == "call" and e.args[0] == "format"


# ---------------------------------------------------------------------------
# Unparse round trip
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "text",
    [
        "success() && github.event_name == 'push'",
        "!(a.b == 'x')",
        "(github.ref == 'a' || github.ref == 'b') && !cancelled()",
        "format('{0}-{1}', matrix.os, 1.5)",
        "contains(fromJSON('[\"a\",\"b\"]'), matrix.x)",
        "github.run_number >= 3 || failure()",
        "fromJSON(steps.a.outputs.json).key[0]",
        "needs['a[x=1]/b'].result == 'success'",
        "'it''s' != null",
    ],
)
def test_unparse_round_trip(text):
    e = X.parse(text)
    assert X.parse(X.unparse(e)) == e


def test_unparse_is_canonical():
    assert X.unparse(X.parse("  github.ref=='x'  ")) == "github.ref == 'x'"
    assert X.unparse(X.parse("(a.b || c.d) && e.f")) == "(a.b || c.d) && e.f"
    assert X.unparse(X.parse("matrix['a b']")) == "matrix['a b']"


# ---------------------------------------------------------------------------
# References and status functions
# ---------------------------------------------------------------------------

def test_refs():
    e = X.parse(
        "steps.a.outputs.x == needs.b.result && matrix.os == github.ref && github.run_number > 1"
    )
    assert X.ctx_refs(e) == frozenset({"matrix.os", "github.ref", "github.run_number"})
    assert X.step_refs(e) == frozenset({"a"})
    assert X.needs_refs(e) == frozenset({"b"})


def test_refs_inside_opaque():
    e = X.parse("steps.a.outputs.n > needs.b.outputs.m")
    assert e.op == "opaque"
    assert X.step_refs(e) == frozenset({"a"})
    assert X.needs_refs(e) == frozenset({"b"})
    assert X.ctx_refs(e) == frozenset()


@pytest.mark.parametrize(
    "text,expected",
    [
        ("success()", True),
        ("failure() && github.ref == 'x'", True),
        ("!cancelled()", True),
        ("always() || github.run_number > 3", True),
        ("github.ref == 'refs/heads/main'", False),
        ("contains(github.ref, 'success()')", False),
        ("steps.x.outcome == 'success'", False),
    ],
)
def test_uses_status_function(text, expected):
    assert X.uses_status_function(X.parse(text)) is expected


# ---------------------------------------------------------------------------
# Substitution and interpolation
# ---------------------------------------------------------------------------

def test_substitute_matrix_values():
    env = {"matrix": {"os": "ubuntu-latest", "node": 18}}
    e = X.substitute(X.parse("matrix.os == 'ubuntu-latest' && matrix.node"), env)
    assert e == Expr("and", (Expr("eq", (lit("ubuntu-latest"), lit("ubuntu-latest"))), lit(18)))


def test_substitute_missing_key_is_null_and_case_insensitive():
    env = {"matrix": {"OS": "x"}}
    assert X.substitute(X.parse("matrix.os"), env) == lit("x")
    assert X.substitute(X.parse("matrix.missing"), env) == lit(None)


def test_substitute_nested_and_expr_values():
    env = {"matrix": {"cfg": {"a": 1}}, "inputs": {"ref": ctx("github.ref")}}
    assert X.substitute(X.parse("matrix.cfg.a"), env) == lit(1)
    assert X.substitute(X.parse("inputs.ref == 'x'"), env) == Expr("eq", (ctx("github.ref"), lit("x")))
    # object values become fromJSON literals
    assert X.substitute(X.parse("toJSON(matrix.cfg)"), env) == Expr(
        "call", ("toJSON", Expr("call", ("fromJSON", lit('{"a":1}'))))
    )


def test_substitute_prefix_keys():
    env = {"secrets.token": ctx("secrets.NPM_TOKEN")}
    assert X.substitute(X.parse("secrets.token"), env) == ctx("secrets.NPM_TOKEN")
    assert X.substitute(X.parse("secrets.other"), env) == ctx("secrets.other")


def test_substitute_inside_opaque_then_fold():
    e = X.parse("matrix.node >= 18")
    assert e.op == "opaque"
    s = X.substitute(e, {"matrix": {"node": 20}})
    assert s == Expr("opaque", ("20 >= 18",))
    assert X.simplify(s, boolean=True) == TRUE


def test_needs_rename():
    e = X.parse("needs.build.result == 'success' && needs.other.result")
    r = X.rename_needs(e, {"build": "call[os=x]/build"})
    assert X.needs_refs(r) == frozenset({"call[os=x]/build", "other"})
    assert X.parse(X.unparse(r)) == r


def test_interpolate():
    env = {"matrix": {"node": 18, "flag": True}}
    assert X.interpolate("node ${{ matrix.node }} on ${{ github.ref }}", env) == "node 18 on ${{ github.ref }}"
    assert X.interpolate("${{ matrix.flag }}", env) == "true"
    # untouched segments are kept verbatim, including spacing
    assert X.interpolate("${{github.sha}}-${{ matrix.node }}", env) == "${{github.sha}}-18"
    # non-constant results are re-rendered
    assert X.interpolate("${{ matrix.node == 18 && github.ref }}", env) == "${{ github.ref }}"
    assert X.interpolate(20, env) == "20"
    assert X.interpolate("${{ 'a}}b' }}", {}) == "${{ 'a}}b' }}"


def test_interpolate_expression_value():
    env = {"inputs": {"tag": Expr("needs", ("prep", "outputs.tag"))}}
    assert X.interpolate("v=${{ inputs.tag }}", env) == "v=${{ needs.prep.outputs.tag }}"


def test_template_segments_respect_quotes():
    segs = X.template_segments("a ${{ 'x}}y' }} b")
    assert [s[0] for s in segs] == [False, True, False]
    assert segs[1][1] == " 'x}}y' "


def test_parse_template():
    assert X.parse_template("plain") == lit("plain")
    assert X.parse_template("${{ matrix.x }}") == ctx("matrix.x")
    t = X.parse_template("v-${{ steps.v.outputs.version }}-{x}")
    assert t == Expr("call", ("format", lit("v-{0}-{{x}}"), Expr("step", ("v", "outputs.version"))))
    assert X.evaluate(t, lambda n: "1.2") == "v-1.2-{x}"
    assert X.parse_template(5) == lit(5)


# ---------------------------------------------------------------------------
# Simplification
# ---------------------------------------------------------------------------

def test_simplify_value_semantics_of_and_or():
    assert X.simplify(X.parse("'a' && 'b'")) == lit("b")
    assert X.simplify(X.parse("'' || 'b'")) == lit("b")
    assert X.simplify(X.parse("0 && github.ref")) == lit(0)
    # value context keeps `x && true` (its value is `true` or falsy x)
    e = X.parse("github.ref && true")
    assert X.simplify(e) == e
    # boolean context only keeps truthiness
    assert X.simplify(e, boolean=True) == ctx("github.ref")
    assert X.simplify(X.parse("github.ref || true"), boolean=True) == TRUE
    assert X.simplify(X.parse("github.ref && false"), boolean=True) == X.FALSE


def test_simplify_folds_default_condition():
    e = X.conj([X.SUCCESS, X.substitute(X.parse("matrix.os == 'UBUNTU-latest'"), {"matrix": {"os": "ubuntu-latest"}})])
    assert X.simplify(e, boolean=True) == X.SUCCESS
    e = X.conj([X.SUCCESS, X.substitute(X.parse("matrix.os == 'windows'"), {"matrix": {"os": "ubuntu-latest"}})])
    assert X.simplify(e, boolean=True) == X.FALSE


def test_simplify_functions():
    assert X.simplify(X.parse("contains(fromJSON('[\"18\",\"20\"]'), 18)")) == lit(True)
    assert X.simplify(X.parse("contains('Hello', 'ell')")) == lit(True)
    assert X.simplify(X.parse("startsWith('refs/tags/v1', 'REFS/TAGS')")) == lit(True)
    assert X.simplify(X.parse("format('{0}-{1}', 'a', 1)")) == lit("a-1")
    assert X.simplify(X.parse("fromJSON('true')")) == lit(True)
    assert X.simplify(X.parse("join(fromJSON('[1,2]'), '+')")) == lit("1+2")
    # not folded: hashFiles, toJSON, non-scalar fromJSON, anything with free refs
    for text in ["hashFiles('**/package-lock.json')", "toJSON('a')", "fromJSON('[1]')", "contains(github.ref, 'x')"]:
        e = X.parse(text)
        assert X.simplify(e) == e


def test_simplify_not():
    assert X.simplify(X.parse("!false")) == lit(True)
    assert X.simplify(X.parse("!!github.ref"), boolean=True) == ctx("github.ref")
    assert X.simplify(X.parse("!!github.ref")) == X.parse("!!github.ref")


def test_conj():
    assert X.conj([]) == TRUE
    assert X.conj([X.SUCCESS]) == X.SUCCESS
    assert X.conj([X.SUCCESS, ctx("a"), ctx("b")]) == Expr("and", (Expr("and", (X.SUCCESS, ctx("a"))), ctx("b")))


# ---------------------------------------------------------------------------
# Evaluation and coercions
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "text,expected",
    [
        ("'18' == 18", True),
        ("'abc' == 'ABC'", True),
        ("'true' == true", False),
        ("null == 0", True),
        ("'' == 0", True),
        ("NaN == NaN", False),
        ("'b' > 'A'", True),
        ("'10' < 9", False),
        ("1 < 'x'", False),
        ("fromJSON('[1]') == fromJSON('[1]')", False),
        ("!''", True),
        ("0 || 'x'", "x"),
    ],
)
def test_evaluate_constants(text, expected):
    assert X.evaluate(X.parse(text)) == expected


def test_evaluate_with_lookup():
    values = {"github.event_name": "push", "success": True}

    def lookup(n: Expr):
        if n.op == "ctx":
            return values[n.args[0]]
        if n.op == "status":
            return values[n.args[0]]
        raise KeyError(n)

    assert X.evaluate(X.parse("success() && github.event_name == 'push'"), lookup) is True
    with pytest.raises(X.Unresolved):
        X.evaluate(X.parse("github.ref == 'x'"), lookup)
    with pytest.raises(X.Unresolved):
        X.evaluate(X.parse("github.ref"))


def test_evaluate_object_filter():
    doc = "fromJSON('{\"labels\":[{\"name\":\"a\"},{\"name\":\"b\"}]}')"
    assert X.evaluate(X.parse(doc + ".labels.*.name")) == ["a", "b"]
    assert X.evaluate(X.parse(doc + ".labels.name")) is None
    assert X.evaluate(X.parse(doc + ".labels[1].name")) == "b"
    assert X.evaluate(X.parse(f"contains({doc}.labels.*.name, 'B')")) is True

    def lookup(n: Expr):
        assert n == ctx("github.event.labels.*.name")
        return ["a", "b"]

    assert X.evaluate(X.parse("contains(github.event.labels.*.name, 'B')"), lookup) is True


def test_to_str_and_truthy():
    assert X.to_str(None) == ""
    assert X.to_str(True) == "true"
    assert X.to_str(3.0) == "3"
    assert X.to_str(3.1) == "3.1"
    assert X.to_str({"b": 1, "a": [1]}) == '{"a":[1],"b":1}'
    assert X.to_str(X.parse("github.ref")) == "${{ github.ref }}"
    assert not X.truthy(0) and not X.truthy("") and not X.truthy(None) and not X.truthy(math.nan)
    assert X.truthy("false") and X.truthy([]) and X.truthy(-1)


def test_to_number():
    assert X.to_number("") == 0.0
    assert X.to_number(" 12 ") == 12.0
    assert math.isnan(X.to_number("12px"))
    assert X.to_number(None) == 0.0
    assert X.to_number(True) == 1.0


def test_constant_value_and_from_value():
    assert X.constant_value(lit(1)) == (True, 1)
    assert X.constant_value(X.from_value([1, "a"])) == (True, [1, "a"])
    assert X.constant_value(ctx("a")) == (False, None)
    assert X.from_value(ctx("a")) == ctx("a")


def test_determinism_of_opaque_refs():
    e1 = X.parse("github.b > 1 && github.a < matrix.z")
    e2 = X.parse("github.b > 1 && github.a < matrix.z")
    assert e1 == e2
    assert e1.args[1].args[1:] == tuple(sorted(e1.args[1].args[1:]))
