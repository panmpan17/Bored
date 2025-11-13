# Usage: if has_arg "argName" "$@"; then
has_arg() {
    local target="$1"
    shift
    for arg in "$@"; do
        if [ "$arg" == "$target" ]; then
            return 0  # found
        fi
    done
    return 1  # not found
}

get_arg() {
    local target="$1"
    shift
    local found=0
    for arg in "$@"; do
        if [ "$found" -eq 1 ]; then
            echo "$arg"
            return 0  # found the parameter
        fi

        if [ "$arg" == "$target" ]; then
            found=1
        fi
    done

    echo ""
    return 1  # not found
}

get_arg_with_default() {
    local target="$1"
    local default="$2"
    shift
    shift
    local found=0
    for arg in "$@"; do
        if [ "$found" -eq 1 ]; then
            echo "$arg"
            return 0  # found the parameter
        fi

        if [ "$arg" == "$target" ]; then
            found=1
        fi
    done

    echo "$default"
    return 1  # not found
}

get_arg_from_choice() {
    local target="$1"
    IFS=" " read -r -a choices <<< "$2"

    shift
    shift
    local found=0
    for arg in "$@"; do
        if [ "$found" -eq 1 ]; then
            for choice in "${choices[@]}"
            do
                if [ "$arg" == "$choice" ]; then
                    echo "$arg"
                    return 0  # found the parameter
                fi
            done

            echo "error"
            return 1
        fi

        if [ "$arg" == "$target" ]; then
            found=1
        fi
    done

    echo ""
    return 1  # not found
}

get_arg_from_choice_with_default() {
    local target="$1"
    IFS=" " read -r -a choices <<< "$2"
    local default="$3"

    shift
    shift
    shift
    local found=0
    for arg in "$@"; do
        if [ "$found" -eq 1 ]; then
            for choice in "${choices[@]}"
            do
                if [ "$arg" == "$choice" ]; then
                    echo "$arg"
                    return 0  # found the parameter
                fi
            done

            echo "error"
            return 0
        fi

        if [ "$arg" == "$target" ]; then
            found=1
        fi
    done

    echo "$default"
    return 1  # not found
}

result=$(get_arg "-f" "$@")
echo "$result"

result=$(get_arg_with_default "-f" "table" "$@")
echo "$result"

result=$(get_arg_from_choice "-f" "table bullet" "$@")
echo "$result"

result=$(get_arg_from_choice_with_default "-f" "table bullet" "table" "$@")
echo "$result"
