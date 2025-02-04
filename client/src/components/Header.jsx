import React from "react";
import { ReactComponent as Logo } from "../assets/images/logo.svg";
import BGImage from "../assets/images/hero-mobile.jpg";

function Header() {
  return (
    <div>
      <Logo className="logo" />
      <img src={BGImage} className="header-img" alt="Hero" />
    </div>
  );
}

export default Header;
