'use client';
import Link from 'next/link';
import { Navbar as BSNavbar, Container, Nav } from 'react-bootstrap';

export default function Navbar() {
    return (
        <BSNavbar expand="lg" className="navbar" variant="dark" fixed="top">
            <Container>
                <Link href="/" passHref legacyBehavior>
                    <BSNavbar.Brand style={{ cursor: 'pointer' }}>
                        <i className="ph ph-barbell me-2"></i>
                        FitGen AI
                    </BSNavbar.Brand>
                </Link>
                <BSNavbar.Toggle aria-controls="navbarNav" />
                <BSNavbar.Collapse id="navbarNav">
                    <Nav className="ms-auto">
                        <Nav.Link as={Link} href="/login">Login</Nav.Link>
                        <Nav.Link as={Link} href="/register">Register</Nav.Link>
                        <Nav.Link as={Link} href="/admin/login">Admin Login</Nav.Link>
                    </Nav>
                </BSNavbar.Collapse>
            </Container>
        </BSNavbar>
    );
}
