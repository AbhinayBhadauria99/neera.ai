import React from "react";
import { ReactComponent as Arrow } from "../assets/images/icon-arrow.svg";
import { useForm } from "react-hook-form";

function Input() {
  const {
    register,
    handleSubmit,
    formState: { errors },
  } = useForm();

  const onSubmit = (values) => alert("Success");

  return (
    <div className="input-container">
      <form onSubmit={handleSubmit(onSubmit)}>
        <input
          autoComplete="off"
          className="input-field"
          type="text"
          placeholder="Email Address"
          {...register("email", {
            required: true,
            pattern: /^[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,4}$/i,
          })}
        />
        <button className="arrow" type="submit">
          <Arrow />
        </button>
      </form>
      {errors.email && <span className="error">Invalid email</span>}
    </div>
  );
}

export default Input;
