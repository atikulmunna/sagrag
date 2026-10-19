"""Storage naming: tenant and domain namespaces can never collide, names are
validated, and isolated searches never fall back to shared data."""

import pytest

import namespaces as ns


def test_tenant_namespace_unreachable_through_a_domain():
    # Regression: tenant "globex" asking for domain "acme" used to read
    # docs_acme, which was tenant acme's private namespace.
    assert ns.target_names("globex", "acme") != ns.target_names("acme", None)
    assert ns.target_names(None, "acme") != ns.target_names("acme", None)


@pytest.mark.parametrize("bad", ["t_acme", "a__b", "../etc", "a b", "-x", "x-", "", None, "a" * 49])
def test_validate_domain_rejects_unsafe_names(bad):
    with pytest.raises(ns.InvalidName):
        ns.validate_domain(bad)


def test_validate_domain_normalizes_case():
    assert ns.validate_domain(" Stoicism ") == "stoicism"


def test_tenant_slug_keeps_slugs_and_hashes_everything_else():
    assert ns.tenant_slug("acme") == "acme"
    hashed = ns.tenant_slug("user@example.com")
    assert hashed.startswith("h-") and ns.is_slug(hashed)
    # Case variants are distinct tenants, and slugging is idempotent.
    assert ns.tenant_slug("Acme") != ns.tenant_slug("acme")
    assert ns.tenant_slug(hashed) == hashed
    # Separator injection is impossible.
    assert "__d_" not in ns.tenant_slug("acme__d_stoicism")


def test_slugify_folder_names():
    assert ns.slugify("Stoic Texts") == "stoic-texts"
    assert ns.slugify("  ") is None
    assert ns.slugify("my_domain") == "my_domain"  # existing docs_my_domain data keeps its name
    assert ns.slugify("a__b--c") == "a_b-c"
    assert ns.slugify("t_private") is None


def test_search_targets_fail_closed_for_tenants():
    assert ns.search_targets("acme", "stoicism") == [
        ("docs_t_acme__d_stoicism", "docs_index_t_acme__d_stoicism")
    ]
    assert ns.search_targets("acme", None) == [("docs_t_acme", "docs_index_t_acme")]
    # Shared layout keeps the legacy base fallback.
    assert ns.search_targets(None, "stoicism") == [
        ("docs_stoicism", "docs_index_stoicism"),
        ("docs", "docs_index"),
    ]


def test_owns_name():
    base = ns.BASE_COLLECTION
    assert ns.owns_name("docs_t_acme", "acme", base)
    assert ns.owns_name("docs_t_acme__d_stoicism", "acme", base)
    assert not ns.owns_name("docs_t_acmeco", "acme", base)
    assert not ns.owns_name("docs_t_acme_x", "acme", base)
    assert not ns.owns_name("docs_stoicism", "acme", base)
    # Shared layout never touches tenant namespaces.
    assert ns.owns_name("docs", None, base)
    assert ns.owns_name("docs_stoicism", None, base)
    assert not ns.owns_name("docs_t_acme", None, base)


def test_domains_from_index_names():
    names = [
        "docs_index",
        "docs_index_stoicism",
        "docs_index_t_acme",
        "docs_index_t_acme__d_finance",
    ]
    assert ns.domains_from_index_names(names, None) == ["stoicism"]
    assert ns.domains_from_index_names(names, "acme") == ["finance"]
    assert ns.domains_from_index_names(names, "globex") == []
