"""
LinkedIn person profile scraping tools.

Uses innerText extraction for resilient profile data capture
with configurable section selection.
"""

import logging
from typing import Annotated, Any

from fastmcp import Context, FastMCP
from fastmcp.exceptions import ToolError
from pydantic import Field

from linkedin_mcp_server.callbacks import MCPContextProgressCallback
from linkedin_mcp_server.config.schema import DEFAULT_TOOL_TIMEOUT_SECONDS
from linkedin_mcp_server.core.exceptions import AuthenticationError
from linkedin_mcp_server.dependencies import get_ready_extractor, handle_auth_error
from linkedin_mcp_server.error_handler import raise_tool_error
from linkedin_mcp_server.scraping import parse_person_sections
from linkedin_mcp_server.scraping.extractor import FilterValidationError

logger = logging.getLogger(__name__)


def register_person_tools(
    mcp: FastMCP, *, tool_timeout: float = DEFAULT_TOOL_TIMEOUT_SECONDS
) -> None:
    """Register all person-related tools with the MCP server."""

    @mcp.tool(
        timeout=tool_timeout,
        title="Get Person Profile",
        annotations={"readOnlyHint": True, "openWorldHint": True},
        tags={"person", "scraping"},
        exclude_args=["extractor"],
    )
    async def get_person_profile(
        linkedin_username: str,
        ctx: Context,
        sections: str | None = None,
        max_scrolls: Annotated[int, Field(ge=1, le=50)] | None = None,
        extractor: Any | None = None,
    ) -> dict[str, Any]:
        """
        Get a specific person's LinkedIn profile.

        Args:
            linkedin_username: LinkedIn username (e.g., "stickerdaniel", "williamhgates"). A full profile URL is accepted too and is reduced to the username.
            ctx: FastMCP context for progress reporting
            sections: Comma-separated list of extra sections to scrape.
                The main profile page is always included.
                Available sections: experience, education, interests, honors, languages, certifications, skills, projects, contact_info, posts
                Examples: "experience,education", "contact_info", "skills,projects", "honors,languages", "posts"
                Default (None) scrapes only the main profile page.
            max_scrolls: Maximum pagination attempts per section to load more content.
                On detail sections (experience, certifications, skills, etc.) this
                is the max number of "Show more" button clicks. On activity/posts
                it is the max scroll-to-bottom iterations. Applies to all sections
                in this call. Default (None) uses 5 for detail sections and 10 for
                posts. Increase when a profile has many items in a section
                (e.g., 30+ certifications, max_scrolls=20). To avoid slowing down
                other sections, request heavy sections in a separate call.

        Returns:
            Dict with url, sections (name -> raw text), and optional references.
            Sections may be absent if extraction yielded no content for that page.
            Includes unknown_sections list when unrecognised names are passed.
            The LLM should parse the raw text in each section.
        """
        try:
            extractor = extractor or await get_ready_extractor(
                ctx, tool_name="get_person_profile"
            )
            requested, unknown = parse_person_sections(sections)

            logger.info(
                "Scraping profile: %s (sections=%s)",
                linkedin_username,
                sections,
            )

            cb = MCPContextProgressCallback(ctx)
            result = await extractor.scrape_person(
                linkedin_username,
                requested,
                callbacks=cb,
                max_scrolls=max_scrolls,
            )

            if unknown:
                result["unknown_sections"] = unknown

            return result

        except AuthenticationError as e:
            try:
                await handle_auth_error(e, ctx)
            except Exception as relogin_exc:
                raise_tool_error(relogin_exc, "get_person_profile")
        except Exception as e:
            raise_tool_error(e, "get_person_profile")  # NoReturn

    @mcp.tool(
        timeout=tool_timeout,
        title="Search People",
        annotations={"readOnlyHint": True, "openWorldHint": True},
        tags={"person", "search"},
        exclude_args=["extractor"],
    )
    async def search_people(
        keywords: str,
        ctx: Context,
        location: str | None = None,
        network: list[str] | None = None,
        current_company: str | None = None,
        extractor: Any | None = None,
    ) -> dict[str, Any]:
        """
        Search for people on LinkedIn.

        Args:
            keywords: Search keywords (e.g., "software engineer", "recruiter at Google")
            ctx: FastMCP context for progress reporting
            location: Optional location filter (e.g., "New York", "Remote")
            network: Optional connection-degree filter. Each element is one of
                "F" (1st-degree), "S" (2nd-degree), "O" (3rd-degree and beyond).
                Example: ["F"] to only return 1st-degree connections.
            current_company: Optional current-employer filter. LinkedIn's
                currentCompany facet only filters on the numeric company URN id
                (e.g. "1115" for SAP); plain company names are accepted by the
                URL but ignored by LinkedIn and return the unfiltered result
                set. Look up a company's URN via get_company_profile -- it is
                exposed under references["about"]. For company-wide employee
                demographics (location/education/function breakdown) plus a
                slug-based lookup, use get_company_employees instead.

        Returns:
            Dict with url, sections (name -> raw text), and optional references.
            The LLM should parse the raw text to extract individual people and their profiles.
        """
        try:
            extractor = extractor or await get_ready_extractor(
                ctx, tool_name="search_people"
            )
            logger.info(
                "Searching people: keywords='%s', location='%s', network=%s, current_company='%s'",
                keywords,
                location,
                network,
                current_company,
            )

            await ctx.report_progress(
                progress=0, total=100, message="Starting people search"
            )

            try:
                result = await extractor.search_people(
                    keywords,
                    location,
                    network=network,
                    current_company=current_company,
                )
            except FilterValidationError as e:
                # Validation messages carry actionable detail; surface
                # them as ToolError so mask_error_details doesn't reduce
                # them to "Error calling tool 'search_people'".
                raise ToolError(str(e)) from e

            await ctx.report_progress(progress=100, total=100, message="Complete")

            return result

        except ToolError:
            # Already a properly formatted client-facing error; do not
            # log it as "Unexpected error" via raise_tool_error.
            raise
        except AuthenticationError as e:
            try:
                await handle_auth_error(e, ctx)
            except Exception as relogin_exc:
                raise_tool_error(relogin_exc, "search_people")
        except Exception as e:
            raise_tool_error(e, "search_people")  # NoReturn

    @mcp.tool(
        timeout=tool_timeout,
        title="Connect With Person",
        annotations={"destructiveHint": True, "openWorldHint": True},
        tags={"person", "actions"},
        exclude_args=["extractor"],
    )
    async def connect_with_person(
        linkedin_username: str,
        ctx: Context,
        note: str | None = None,
        extractor: Any | None = None,
    ) -> dict[str, Any]:
        """
        Send a LinkedIn connection request or accept an incoming one.

        The tool is annotated with destructiveHint so MCP clients will
        prompt for user confirmation before execution.

        Args:
            linkedin_username: LinkedIn username (e.g., "stickerdaniel", "williamhgates"). A full profile URL is accepted too and is reduced to the username.
            ctx: FastMCP context for progress reporting
            note: Optional note to include with the invitation

        Returns:
            Dict with url, status, message, and note_sent.
            Statuses: pending, already_connected, follow_only,
            connect_unavailable, unavailable, send_failed,
            note_not_supported, custom_note_limit_reached,
            connected, or accepted.

            When status is ``custom_note_limit_reached`` LinkedIn rejected
            personalized invite notes because the free note quota for the
            account is exhausted. The ``message`` is the raw Premium dialog
            text read from LinkedIn.
        """
        try:
            extractor = extractor or await get_ready_extractor(
                ctx, tool_name="connect_with_person"
            )
            logger.info(
                "Connecting with person: %s (note=%s)",
                linkedin_username,
                note is not None,
            )

            await ctx.report_progress(
                progress=0,
                total=100,
                message="Starting LinkedIn connection flow",
            )

            result = await extractor.connect_with_person(
                linkedin_username,
                note=note,
            )

            await ctx.report_progress(progress=100, total=100, message="Complete")

            return result

        except AuthenticationError as e:
            try:
                await handle_auth_error(e, ctx)
            except Exception as relogin_exc:
                raise_tool_error(relogin_exc, "connect_with_person")
        except Exception as e:
            raise_tool_error(e, "connect_with_person")  # NoReturn

    @mcp.tool(
        timeout=tool_timeout,
        title="Get Sidebar Profiles",
        annotations={"readOnlyHint": True, "openWorldHint": True},
        tags={"person", "scraping"},
        exclude_args=["extractor"],
    )
    async def get_sidebar_profiles(
        linkedin_username: str,
        ctx: Context,
        extractor: Any | None = None,
    ) -> dict[str, Any]:
        """
        Get profile links from sidebar recommendation sections on a LinkedIn profile page.

        Extracts profiles from "More profiles for you", "Explore premium profiles",
        and "People you may know" sidebar sections. Follows "Show all" links to
        return the full list from each section. Sections that redirect to
        linkedin.com/premium are skipped.

        Args:
            linkedin_username: LinkedIn username of the profile page to scrape; a full profile URL is accepted too
                (e.g., "stickerdaniel", "williamhgates")
            ctx: FastMCP context for progress reporting

        Returns:
            Dict with url and sidebar_profiles mapping section key to a list of
            /in/username/ paths. Only sections present on the page are included.
        """
        try:
            extractor = extractor or await get_ready_extractor(
                ctx, tool_name="get_sidebar_profiles"
            )
            logger.info("Getting sidebar profiles for: %s", linkedin_username)

            await ctx.report_progress(
                progress=0, total=100, message="Extracting sidebar profiles"
            )

            result = await extractor.get_sidebar_profiles(linkedin_username)

            await ctx.report_progress(progress=100, total=100, message="Complete")

            return result

        except AuthenticationError as e:
            try:
                await handle_auth_error(e, ctx)
            except Exception as relogin_exc:
                raise_tool_error(relogin_exc, "get_sidebar_profiles")
        except Exception as e:
            raise_tool_error(e, "get_sidebar_profiles")  # NoReturn

    @mcp.tool(
        timeout=tool_timeout,
        title="Get Mutual Connections",
        annotations={"readOnlyHint": True, "openWorldHint": True},
        tags={"person", "scraping"},
        exclude_args=["extractor"],
    )
    async def get_mutual_connections(
        linkedin_username: str,
        ctx: Context,
        keywords: str | None = None,
        max_scrolls: Annotated[int, Field(ge=1, le=50)] = 5,
        extractor: Any | None = None,
    ) -> dict[str, Any]:
        """
        List the connections you share with a person -- who you both know.

        Answers "who could introduce me to this person", which the company-based
        filters cannot: search_people needs a company URN, so it only finds warm
        paths at an employer you already identified. This works from the person.

        Works for 1st- and 2nd-degree profiles alike -- you share connections
        with a direct contact too. Your own profile has nothing to share, and
        anywhere the member id cannot be read the result comes back empty with
        a ``no_mutual_connections_link`` section error rather than a people
        search filtered to the wrong thing.

        An empty result usually means no shared connections, not a failure.
        LinkedIn only renders the shared-connections link when a shared set
        exists, and that link is the only thing this tool will follow -- it
        will not reconstruct the search from a member id. A reconstructed url
        was tried and removed: LinkedIn accepted it, silently dropped the
        ``connectionOf`` facet, and returned the caller's entire 1st-degree
        network, which reads exactly like a real answer. Same class of bug as
        the ``currentCompany`` caveat in ``search_people``.

        So do not retry an empty result hoping for a list, and never report a
        warm introduction on the strength of one. To confirm independently,
        read the target with ``get_person_profile``: a real shared connection
        shows on the top card as "<Name> is a mutual connection" with a
        matching ``mutual_connections`` reference. No line means none exist.

        Note the subtitle on each returned card ("X, Y & N other mutual
        connections") describes YOUR overlap with that person, not theirs with
        the target. It is not a check on this tool's own filtering.

        Costs two page loads, so prefer it for named targets over sweeping a list.

        Args:
            linkedin_username: LinkedIn username of the person; a full profile URL is accepted too
                (e.g., "stickerdaniel", "williamhgates")
            ctx: FastMCP context for progress reporting
            keywords: Optional filter over the shared connections, by name, title or skill
                (e.g., "recruiter", "engineer")
            max_scrolls: How far to scroll the results page (1-50, default 5)

        Returns:
            Dict with url, sections {mutual_connections: raw text}, references
            (/in/ paths for the shared connections) and optional section_errors.
        """
        try:
            extractor = extractor or await get_ready_extractor(
                ctx, tool_name="get_mutual_connections"
            )
            logger.info("Getting mutual connections for: %s", linkedin_username)

            await ctx.report_progress(
                progress=0, total=100, message="Extracting mutual connections"
            )

            result = await extractor.get_mutual_connections(
                linkedin_username, keywords=keywords, max_scrolls=max_scrolls
            )

            await ctx.report_progress(progress=100, total=100, message="Complete")

            return result

        except AuthenticationError as e:
            try:
                await handle_auth_error(e, ctx)
            except Exception as relogin_exc:
                raise_tool_error(relogin_exc, "get_mutual_connections")
        except Exception as e:
            raise_tool_error(e, "get_mutual_connections")  # NoReturn

    @mcp.tool(
        timeout=tool_timeout,
        title="Find Warm Paths At Company",
        annotations={"readOnlyHint": True, "openWorldHint": True},
        tags={"person", "company", "scraping"},
        exclude_args=["extractor"],
    )
    async def find_warm_paths_at_company(
        company_urn: str,
        ctx: Context,
        keywords: str | None = None,
        network: list[str] | None = None,
        max_people: Annotated[int, Field(ge=1, le=25)] = 5,
        max_scrolls: Annotated[int, Field(ge=1, le=50)] | None = None,
        extractor: Any | None = None,
    ) -> dict[str, Any]:
        """Find who at a company you share connections with, and name the shared
        connections.

        The warm-intro sweep. get_mutual_connections answers "who do we both
        know" for one person you already picked; this finds the people worth
        picking, across a whole employer, and answers it for each of them in the
        same pass. Use it when targeting a company rather than an individual.

        Each returned entry carries the employee (name, profile_url, headline),
        LinkedIn's own summary line ("Jack is a mutual connection"), and
        ``mutuals``: the people you actually share, by name. Those are who can
        make the introduction.

        Cost and volume. One page load for the employee search, then one per
        person expanded, bounded by ``max_people`` (default 5). People without a
        shared-connections anchor cost nothing -- they are skipped, not visited.
        Raise ``max_people`` deliberately; every increment is another scrape
        against your own account.

        Absence is meaningful, not an error. LinkedIn renders the anchor only
        where a shared set exists, and this follows anchors verbatim rather than
        constructing searches. An empty ``warm_paths`` means no warm paths among
        the employees on that results page. It never means "retry".

        Args:
            company_urn: Numeric LinkedIn company URN id (e.g. "75527963").
                get_company_employees returns it as a ``company_urn`` reference;
                so does get_company_profile. A company name or slug is refused,
                because LinkedIn silently ignores non-numeric values here and
                would return an unfiltered people search.
            keywords: Optional filter over employees ("recruiter", "chief of
                staff", "go to market"). Narrow first when a company is large --
                a results page holds a limited number of cards.
            network: Connection-degree filter, defaults to ["F", "S"]. 3rd-degree
                profiles are excluded by default: LinkedIn renders no
                shared-connections anchor for them, so they only crowd out the
                page.
            max_people: How many people to expand, 1-25, default 5.
            max_scrolls: Scroll depth on the employee search page.

        Returns:
            Dict with url, company_urn, warm_paths, people_with_mutuals_found,
            people_expanded, and optionally truncated/note/section_errors.
        """
        try:
            extractor = extractor or await get_ready_extractor(
                ctx, tool_name="find_warm_paths_at_company"
            )
            logger.info("Finding warm paths at company: %s", company_urn)

            await ctx.report_progress(
                progress=0, total=100, message="Searching employees"
            )

            result = await extractor.find_warm_paths_at_company(
                company_urn,
                keywords=keywords,
                network=network,
                max_people=max_people,
                max_scrolls=max_scrolls,
            )

            await ctx.report_progress(progress=100, total=100, message="Complete")

            return result

        except AuthenticationError as e:
            try:
                await handle_auth_error(e, ctx)
            except Exception as relogin_exc:
                raise_tool_error(relogin_exc, "find_warm_paths_at_company")
        except Exception as e:
            raise_tool_error(e, "find_warm_paths_at_company")  # NoReturn

    @mcp.tool(
        timeout=tool_timeout,
        title="Get My Profile",
        annotations={"readOnlyHint": True, "openWorldHint": True},
        tags={"person", "scraping"},
        exclude_args=["extractor"],
    )
    async def get_my_profile(
        ctx: Context,
        sections: str | None = None,
        max_scrolls: Annotated[int, Field(ge=1, le=50)] | None = None,
        extractor: Any | None = None,
    ) -> dict[str, Any]:
        """
        Get the authenticated user's own LinkedIn profile.

        Navigates to /in/me/ and resolves the redirect to obtain the real
        username before scraping, so the url field in the result is the actual
        profile URL (e.g. linkedin.com/in/johndoe/) rather than /in/me/.

        Args:
            ctx: FastMCP context for progress reporting
            sections: Comma-separated list of extra sections to scrape.
                The main profile page is always included.
                Available sections: experience, education, interests, honors, languages, certifications, skills, projects, contact_info, posts
                Examples: "experience,education", "contact_info", "skills,projects"
                Default (None) scrapes only the main profile page.
            max_scrolls: Maximum pagination attempts per section (same as get_person_profile).

        Returns:
            Dict with url, sections (name -> raw text), and optional references.
            The url field reflects the resolved profile URL, revealing the real username.
        """
        try:
            extractor = extractor or await get_ready_extractor(
                ctx, tool_name="get_my_profile"
            )
            requested, unknown = parse_person_sections(sections)

            logger.info("Scraping own profile (sections=%s)", sections)

            cb = MCPContextProgressCallback(ctx)
            result = await extractor.get_my_profile(
                sections=requested,
                callbacks=cb,
                max_scrolls=max_scrolls,
            )

            if unknown:
                result["unknown_sections"] = unknown

            return result

        except AuthenticationError as e:
            try:
                await handle_auth_error(e, ctx)
            except Exception as relogin_exc:
                raise_tool_error(relogin_exc, "get_my_profile")
        except Exception as e:
            raise_tool_error(e, "get_my_profile")  # NoReturn
